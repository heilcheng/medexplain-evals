//! Fetch MedlinePlus health topic summaries (NLM, public domain per NLM policy).
//!
//! Only the health topic summary text is used. Drug monographs (ASHP), the A.D.A.M.
//! encyclopedia and supplement monographs are copyrighted and are never fetched.

use crate::text::{clean, truncate_words};
use crate::{PipelineError, Result};
use plainline_core::schema::{CANARY, Source, SourceKind, Split};
use regex::Regex;
use std::collections::{BTreeSet, HashMap};
use std::time::Duration;

pub const INDEX: &str = "https://medlineplus.gov/xml.html";
pub const XML_BASE: &str = "https://medlineplus.gov/xml";

#[derive(Debug, Clone)]
pub struct Topic {
    pub title: String,
    pub url: String,
    pub summary: String,
}

/// Name of the newest topics file listed on the index page.
pub fn latest_topics_name(index_html: &str) -> Result<String> {
    let re = Regex::new(r"mplus_topics_\d{4}-\d{2}-\d{2}\.xml").unwrap();
    let names: BTreeSet<&str> = re.find_iter(index_html).map(|m| m.as_str()).collect();
    names
        .last()
        .map(|s| s.to_string())
        .ok_or_else(|| PipelineError::Data("could not find a MedlinePlus topics XML file".into()))
}

/// Map lowercase English title to the topic.
pub fn parse_topics(xml: &str) -> Result<HashMap<String, Topic>> {
    let opts = roxmltree::ParsingOptions {
        allow_dtd: true,
        ..Default::default()
    };
    let doc = roxmltree::Document::parse_with_options(xml, opts)?;
    let mut out = HashMap::new();
    for ht in doc
        .descendants()
        .filter(|n| n.is_element() && n.tag_name().name() == "health-topic")
    {
        if ht.attribute("language") != Some("English") {
            continue;
        }
        let Some(fs) = ht
            .children()
            .find(|c| c.is_element() && c.tag_name().name() == "full-summary")
        else {
            continue;
        };
        // ElementTree's `.text`: the text before the first child element.
        let text = fs
            .first_child()
            .filter(|c| c.is_text())
            .and_then(|c| c.text())
            .unwrap_or("");
        if text.trim().is_empty() {
            continue;
        }
        let title = ht.attribute("title").unwrap_or("").to_string();
        out.insert(
            title.to_lowercase(),
            Topic {
                title,
                url: ht.attribute("url").unwrap_or("").to_string(),
                summary: clean(text),
            },
        );
    }
    Ok(out)
}

pub fn build_source(topic: &Topic, xml_name: &str, fetched: &str) -> Source {
    let slug = Regex::new(r"[^a-z0-9]+")
        .unwrap()
        .replace_all(&topic.title.to_lowercase(), "-")
        .trim_matches('-')
        .to_string();
    Source {
        source_id: format!("mp-{slug}"),
        kind: SourceKind::MedlineplusTopic,
        title: topic.title.clone(),
        text: format!("{} (MedlinePlus health topic summary)\n\n{}", topic.title, truncate_words(&topic.summary, 350)),
        origin_url: topic.url.clone(),
        license_note: "NLM health topic summary, public domain (NLM copyright policy). Courtesy of the U.S. National Library of Medicine."
            .to_string(),
        fetched: fetched.to_string(),
        version: xml_name.to_string(),
        split: Split::Dev,
        canary: CANARY.to_string(),
    }
}

/// Fetch the newest topics file. Returns the sources found and the titles that were missing.
pub async fn fetch_topics(
    client: &reqwest::Client,
    index_url: &str,
    xml_base: &str,
    titles: &[&str],
    fetched: &str,
) -> Result<(Vec<Source>, Vec<String>)> {
    let page = client.get(index_url).send().await?.text().await?;
    let name = latest_topics_name(&page)?;
    let r = client
        .get(format!("{xml_base}/{name}"))
        .timeout(Duration::from_secs(120))
        .send()
        .await?
        .error_for_status()?;
    let topics = parse_topics(&r.text().await?)?;
    let (mut got, mut missing) = (Vec::new(), Vec::new());
    for t in titles {
        match topics.get(&t.to_lowercase()) {
            Some(topic) => got.push(build_source(topic, &name, fetched)),
            None => missing.push(t.to_string()),
        }
    }
    Ok((got, missing))
}
