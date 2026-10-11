//! Gemini (cloud) speech recognition backend.
//!
//! Each VAD speech chunk is sent to the Gemini API as a WAV clip. Timing comes
//! from VAD (word timings are interpolated across the chunk); Gemini only
//! supplies the text. That split keeps subtitle timing as tight as the local
//! engines while letting the prompt control the output, e.g. code-mixed speech
//! written in each word's own script ("මට Door එක අරගන්න ඕන").
//!
//! Needs a `GEMINI_API_KEY` environment variable. `GEMINI_MODEL` overrides the
//! model id.

use crate::types::{LabeledProgressFn, NewSegmentFn, ProgressType, Segment, SegmentStage, SpeechSegment, TranscribeOptions};
use crate::utils::{interpolate_word_timestamps, push_segment_clamped, split_speech_segment};
use base64::Engine as _;
use eyre::{bail, eyre, Result};
use serde_json::{json, Value};
use std::time::Duration;

const DEFAULT_MODEL: &str = "gemini-3.5-flash";
// Short clips keep each request fast and the interpolated word timings close.
const MAX_SEGMENT_SECONDS: f64 = 30.0;
const MAX_ATTEMPTS: u32 = 5;

fn build_prompt(options: &TranscribeOptions) -> String {
    let mut prompt = String::from(
        "Transcribe the speech in this audio clip exactly as spoken. Do not translate.\n\
         Write every word in the script of the language it was spoken in: when a speaker \
         mixes languages, keep each word in its own script. For example, Sinhala mixed \
         with English is written as: මට Door එක අරගන්න ඕන\n\
         Output only the transcript text on a single line, with no labels, quotes, \
         timestamps or notes. If there is no speech, output nothing.",
    );
    if let Some(lang) = options.lang.as_deref().filter(|l| *l != "auto") {
        prompt.push_str(&format!("\nThe main spoken language is '{lang}' (ISO 639-1)."));
    }
    // The app's "Vocabulary & Context" box (Whisper's initial prompt) doubles as a glossary here.
    if let Some(ctx) = options.advanced.as_ref().and_then(|a| a.init_prompt.as_deref()) {
        prompt.push_str(&format!("\nVocabulary and context: {ctx}"));
    }
    prompt
}

fn encode_wav(samples: &[i16]) -> Result<Vec<u8>> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 16000,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut buf = Vec::new();
    let mut writer = hound::WavWriter::new(std::io::Cursor::new(&mut buf), spec)?;
    for &s in samples {
        writer.write_sample(s)?;
    }
    writer.finalize()?;
    Ok(buf)
}

/// Joins the non-thought text parts of the first candidate. A candidate with no
/// text (blocked chunk, no speech) yields an empty string so one bad chunk does
/// not fail the whole job.
fn response_text(body: &Value) -> String {
    body["candidates"][0]["content"]["parts"]
        .as_array()
        .map(|parts| {
            parts
                .iter()
                .filter(|p| !p["thought"].as_bool().unwrap_or(false))
                .filter_map(|p| p["text"].as_str())
                .collect::<String>()
        })
        .unwrap_or_default()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

async fn transcribe_clip(client: &reqwest::Client, url: &str, api_key: &str, prompt: &str, wav: &[u8]) -> Result<String> {
    let body = json!({
        "contents": [{
            "parts": [
                { "inline_data": { "mime_type": "audio/wav", "data": base64::engine::general_purpose::STANDARD.encode(wav) } },
                { "text": prompt }
            ]
        }]
    })
    .to_string();

    let mut last_err = String::new();
    for attempt in 0..MAX_ATTEMPTS {
        if attempt > 0 {
            tokio::time::sleep(Duration::from_millis(1000 << attempt)).await;
        }
        let request = client
            .post(url)
            .header("x-goog-api-key", api_key)
            .header("content-type", "application/json")
            .body(body.clone());
        let resp = match request.send().await {
            Ok(r) => r,
            Err(e) => {
                last_err = format!("network error: {e}");
                continue;
            }
        };
        let status = resp.status();
        let text = resp.text().await.unwrap_or_default();
        if status.is_success() {
            let value: Value = serde_json::from_str(&text).map_err(|e| eyre!("Gemini returned invalid JSON: {e}"))?;
            return Ok(response_text(&value));
        }
        let message = serde_json::from_str::<Value>(&text)
            .ok()
            .and_then(|v| v["error"]["message"].as_str().map(str::to_string))
            .unwrap_or(text);
        last_err = format!("HTTP {status}: {message}");
        // Only rate limits and server errors are worth retrying; a bad key or request is not.
        if status.as_u16() != 429 && !status.is_server_error() {
            break;
        }
    }
    Err(eyre!("Gemini transcription failed ({last_err})"))
}

pub async fn transcribe_gemini(
    speech_segments: Vec<SpeechSegment>,
    options: &TranscribeOptions,
    progress: Option<&LabeledProgressFn>,
    new_segment: Option<&NewSegmentFn>,
    abort: Option<Box<dyn Fn() -> bool + Send + Sync>>,
) -> Result<(Vec<Segment>, Option<String>)> {
    let cancelled = || abort.as_ref().map(|c| c()).unwrap_or(false);

    let api_key = std::env::var("GEMINI_API_KEY")
        .ok()
        .map(|k| k.trim().to_string())
        .filter(|k| !k.is_empty())
        .ok_or_else(|| eyre!("Gemini needs an API key. Set the GEMINI_API_KEY environment variable, then restart AutoSubs."))?;
    let model = std::env::var("GEMINI_MODEL").ok().filter(|m| !m.trim().is_empty()).unwrap_or_else(|| DEFAULT_MODEL.to_string());
    let url = format!("https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent", model.trim());
    let client = reqwest::Client::builder().timeout(Duration::from_secs(120)).build()?;
    let prompt = build_prompt(options);
    let user_offset = options.offset.unwrap_or(0.0);

    let expanded: Vec<SpeechSegment> = speech_segments
        .iter()
        .flat_map(|seg| split_speech_segment(seg, MAX_SEGMENT_SECONDS))
        .collect();
    let total = expanded.len().max(1);
    let mut segments: Vec<Segment> = Vec::new();

    for (i, speech_segment) in expanded.iter().enumerate() {
        if cancelled() {
            bail!("Transcription cancelled");
        }

        let wav = encode_wav(&speech_segment.samples)?;
        let text = transcribe_clip(&client, &url, &api_key, &prompt, &wav).await?;

        if !text.is_empty() {
            let start = speech_segment.start + user_offset;
            let end = speech_segment.end + user_offset;
            let words = interpolate_word_timestamps(&text, start, end);
            let segment = Segment {
                speaker_id: speech_segment.speaker_id.clone(),
                start,
                end,
                text,
                words: (!words.is_empty()).then_some(words),
            };
            if let Some(cb) = new_segment {
                cb(i, &segment, SegmentStage::Transcribe);
            }
            push_segment_clamped(&mut segments, segment);
        }

        if let Some(cb) = progress {
            let pct = ((i + 1) as f64 / total as f64 * 100.0) as i32;
            cb(pct, ProgressType::Transcribe, "progressSteps.transcribe");
        }
    }

    let lang = options.lang.clone().filter(|l| l != "auto");
    Ok((segments, lang))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn response_text_skips_thoughts_and_joins_parts() {
        let body = json!({"candidates": [{"content": {"parts": [
            {"text": "thinking...", "thought": true},
            {"text": "මට Door එක\n"},
            {"text": " අරගන්න ඕන"}
        ]}}]});
        assert_eq!(response_text(&body), "මට Door එක අරගන්න ඕන");
        assert_eq!(response_text(&json!({"promptFeedback": {"blockReason": "OTHER"}})), "");
    }

    #[test]
    fn wav_has_header_and_samples() {
        let wav = encode_wav(&[0, 1, -1]).unwrap();
        assert_eq!(&wav[..4], b"RIFF");
        assert_eq!(wav.len(), 44 + 3 * 2);
    }

    #[test]
    fn prompt_includes_language_and_vocabulary() {
        let mut options = TranscribeOptions { lang: Some("si".into()), ..Default::default() };
        options.advanced = Some(crate::types::AdvancedTranscribe { init_prompt: Some("AutoSubs".into()), ..Default::default() });
        let prompt = build_prompt(&options);
        assert!(prompt.contains("'si'") && prompt.contains("AutoSubs"));
        assert!(!build_prompt(&TranscribeOptions::default()).contains("ISO 639-1"));
    }
}
