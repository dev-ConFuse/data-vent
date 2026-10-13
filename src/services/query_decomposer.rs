use regex::Regex;
use std::collections::HashSet;
use tracing::info;

#[derive(Debug, Clone)]
pub struct QueryChunk {
    pub text: String,
    pub intent: String,
    pub weight: f64,
    pub original_span: (usize, usize),
    pub tokens: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct DecompositionResult {
    pub _original_query: String,
    pub chunks: Vec<QueryChunk>,
    pub _total_chunks: usize,
    pub decomposition_time_ms: f64,
}

pub struct QueryDecomposer {
    max_chunks: usize,
    min_chunk_length: usize,
    max_ngram_size: usize,
    stop_words: HashSet<String>,
    relationship_keywords: HashSet<String>,
    attribute_keywords: HashSet<String>,
}

impl QueryDecomposer {
    pub fn new(max_chunks: usize) -> Self {
        let stop_words = [
            "a", "an", "the", "this", "that", "these", "those",
            "in", "on", "at", "to", "for", "of", "with", "by", "from", "into", "about",
            "between", "through", "during", "before", "after", "above", "below", "up",
            "down", "out", "off", "over", "under", "and", "or", "but", "nor", "so", "yet",
            "both", "either", "neither", "i", "me", "my", "we", "us", "our", "you",
            "your", "he", "him", "his", "she", "her", "it", "its", "they", "them",
            "their", "who", "whom", "is", "am", "are", "was", "were", "be", "been",
            "being", "has", "have", "had", "do", "does", "did", "will", "would",
            "shall", "should", "may", "might", "can", "could", "what", "how", "where",
            "when", "why", "which", "show", "tell", "give", "get", "find", "search",
            "please", "help", "need", "want", "like", "all", "any", "some", "each",
            "every", "no", "not", "just", "only", "also", "very", "really", "quite",
            "more", "most", "much", "many", "few", "less", "least", "then", "than",
            "too", "here", "there", "now"
        ].iter().map(|&s| s.to_string()).collect();

        let relationship_keywords = [
            "related", "connects", "linked", "associated", "depends", "references",
            "uses", "calls", "inherits", "implements", "contains", "belongs", "owns",
            "maps", "extends", "imports", "exports", "requires", "provides", "between",
            "relationship", "connection", "dependency", "parent", "child", "sibling",
            "ancestor", "descendant"
        ].iter().map(|&s| s.to_string()).collect();

        let attribute_keywords = [
            "type", "name", "value", "status", "state", "count", "size", "length",
            "format", "version", "date", "time", "created", "updated", "modified",
            "deleted", "config", "configuration", "setting", "parameter", "property",
            "attribute", "field", "column", "description", "summary", "title", "label"
        ].iter().map(|&s| s.to_string()).collect();

        Self {
            max_chunks,
            min_chunk_length: 2,
            max_ngram_size: 3,
            stop_words,
            relationship_keywords,
            attribute_keywords,
        }
    }

    #[allow(dead_code)]
    pub async fn decompose_fast(&self, query: &str) -> DecompositionResult {
        let trimmed = query.trim();
        if trimmed.is_empty() {
            return DecompositionResult {
                _original_query: query.to_string(),
                chunks: vec![],
                _total_chunks: 0,
                decomposition_time_ms: 0.0,
            };
        }

        if trimmed.len() < 50 {
            let tokens: Vec<String> = trimmed.split_whitespace().map(|s| s.to_string()).collect();
            return DecompositionResult {
                _original_query: query.to_string(),
                chunks: vec![QueryChunk {
                    text: trimmed.to_string(),
                    intent: "simple".to_string(),
                    weight: 1.0,
                    original_span: (0, trimmed.len()),
                    tokens,
                }],
                _total_chunks: 1,
                decomposition_time_ms: 0.05,
            };
        }

        self.decompose(query).await
    }

    pub async fn decompose(&self, query: &str) -> DecompositionResult {
        let start = std::time::Instant::now();
        if query.trim().is_empty() {
            return DecompositionResult {
                _original_query: query.to_string(),
                chunks: vec![],
                _total_chunks: 0,
                decomposition_time_ms: 0.0,
            };
        }

        let cleaned = self.preprocess(query);
        let (mut all_chunks, remaining) = self.extract_entities(&cleaned, query);
        let ngrams = self.extract_ngrams(&remaining);

        for (text, span, tokens) in ngrams {
            let intent = self.classify_intent(&tokens);
            all_chunks.push(QueryChunk {
                text,
                intent,
                weight: 0.0,
                original_span: span,
                tokens,
            });
        }

        self.assign_weights(&mut all_chunks, query);
        let mut deduplicated = self.deduplicate(all_chunks);
        deduplicated.sort_by(|a, b| b.weight.partial_cmp(&a.weight).unwrap_or(std::cmp::Ordering::Equal));
        deduplicated.truncate(self.max_chunks);

        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;

        info!("query_decomposed original=\"{}\" chunks={} time_ms={:.2}", 
              query.chars().take(100).collect::<String>(), 
              deduplicated.len(), 
              elapsed_ms);

        DecompositionResult {
            _original_query: query.to_string(),
            _total_chunks: deduplicated.len(),
            chunks: deduplicated,
            decomposition_time_ms: elapsed_ms,
        }
    }

    fn preprocess(&self, query: &str) -> String {
        let re_ws = Regex::new(r"\s+").expect("static regex compiles");
        let re_chars = Regex::new(r"[^\w\s\u{0022}'\-_./]").expect("static regex compiles");
        let s1 = re_ws.replace_all(query.trim(), " ").to_string();
        let s2 = re_chars.replace_all(&s1, " ").to_string();
        re_ws.replace_all(&s2, " ").trim().to_string()
    }

    fn extract_entities(&self, cleaned: &str, original: &str) -> (Vec<QueryChunk>, String) {
        let mut chunks = vec![];
        let mut remaining = cleaned.to_string();

        let re_quoted = Regex::new(r#""([^"]+)""#).expect("static regex compiles");
        let mut to_remove = vec![];
        for cap in re_quoted.captures_iter(&remaining) {
            let text = cap[1].trim();
            if text.len() >= self.min_chunk_length {
                let start_pos = original.to_lowercase().find(&text.to_lowercase()).unwrap_or(0);
                chunks.push(QueryChunk {
                    text: text.to_string(),
                    intent: "entity_lookup".to_string(),
                    weight: 1.0,
                    original_span: (start_pos, start_pos + text.len()),
                    tokens: text.split_whitespace().map(|s| s.to_string()).collect(),
                });
                to_remove.push(cap[0].to_string());
            }
        }
        for rem in to_remove {
            remaining = remaining.replace(&rem, " ");
        }

        let re_ident = Regex::new(r"\b([a-zA-Z][a-zA-Z0-9]*(?:[._][a-zA-Z][a-zA-Z0-9]*)+)\b").expect("static regex compiles");
        let mut to_remove2 = vec![];
        for cap in re_ident.captures_iter(&remaining) {
            let text = &cap[1];
            if text.len() >= self.min_chunk_length {
                let start_pos = original.to_lowercase().find(&text.to_lowercase()).unwrap_or(0);
                chunks.push(QueryChunk {
                    text: text.to_string(),
                    intent: "entity_lookup".to_string(),
                    weight: 0.95,
                    original_span: (start_pos, start_pos + text.len()),
                    tokens: vec![text.to_string()],
                });
                to_remove2.push(text.to_string());
            }
        }
        for rem in to_remove2 {
            remaining = remaining.replace(&rem, " ");
        }

        let re_const = Regex::new(r"\b([A-Z][A-Z0-9_]{2,})\b").expect("static regex compiles");
        let mut to_remove3 = vec![];
        for cap in re_const.captures_iter(&remaining) {
            let text = &cap[1];
            let start_pos = original.find(text).unwrap_or(0);
            chunks.push(QueryChunk {
                text: text.to_string(),
                intent: "entity_lookup".to_string(),
                weight: 0.9,
                original_span: (start_pos, start_pos + text.len()),
                tokens: vec![text.to_string()],
            });
            to_remove3.push(text.to_string());
        }
        for rem in to_remove3 {
            remaining = remaining.replace(&rem, " ");
        }

        let re_ws = Regex::new(r"\s+").expect("static regex compiles");
        remaining = re_ws.replace_all(&remaining, " ").trim().to_string();

        (chunks, remaining)
    }

    fn extract_ngrams(&self, text: &str) -> Vec<(String, (usize, usize), Vec<String>)> {
        let words: Vec<&str> = text.split_whitespace().collect();
        let mut meaningful = vec![];
        let re_strip = Regex::new(r"^[^\w]+|[^\w]+$").expect("static regex compiles");

        for (i, word) in words.iter().enumerate() {
            let clean = re_strip.replace_all(word, "").to_string().to_lowercase();
            if !clean.is_empty() && !self.stop_words.contains(&clean) && clean.len() >= self.min_chunk_length {
                meaningful.push((clean, i));
            }
        }

        if meaningful.is_empty() {
            return vec![];
        }

        let mut ngrams = vec![];
        let mut used = HashSet::new();

        if self.max_ngram_size >= 3 && meaningful.len() >= 3 {
            for i in 0..meaningful.len() - 2 {
                let (_, idx1) = meaningful[i];
                let (_, idx2) = meaningful[i + 1];
                let (_, idx3) = meaningful[i + 2];
                if idx2 - idx1 <= 2 && idx3 - idx2 <= 2 {
                    let tokens = vec![meaningful[i].0.clone(), meaningful[i+1].0.clone(), meaningful[i+2].0.clone()];
                    ngrams.push((tokens.join(" "), (idx1, idx3), tokens));
                    used.insert(i);
                    used.insert(i+1);
                    used.insert(i+2);
                }
            }
        }

        if self.max_ngram_size >= 2 && meaningful.len() >= 2 {
            for i in 0..meaningful.len() - 1 {
                if used.contains(&i) && used.contains(&(i + 1)) {
                    continue;
                }
                let (_, idx1) = meaningful[i];
                let (_, idx2) = meaningful[i + 1];
                if idx2 - idx1 <= 2 {
                    let tokens = vec![meaningful[i].0.clone(), meaningful[i+1].0.clone()];
                    ngrams.push((tokens.join(" "), (idx1, idx2), tokens));
                    used.insert(i);
                    used.insert(i+1);
                }
            }
        }

        for (i, (word, idx)) in meaningful.into_iter().enumerate() {
            if !used.contains(&i) {
                ngrams.push((word.clone(), (idx, idx), vec![word]));
            }
        }

        ngrams
    }

    fn classify_intent(&self, tokens: &[String]) -> String {
        for t in tokens {
            let lower = t.to_lowercase();
            if self.relationship_keywords.contains(&lower) {
                return "relationship_query".to_string();
            }
            if self.attribute_keywords.contains(&lower) {
                return "attribute_search".to_string();
            }
        }
        "entity_lookup".to_string()
    }

    fn assign_weights(&self, chunks: &mut [QueryChunk], original: &str) {
        let q_len = original.len().max(1) as f64;
        for chunk in chunks.iter_mut() {
            if chunk.weight > 0.0 {
                continue;
            }
            let mut w = 0.5;
            let t_count = chunk.tokens.len();
            if t_count >= 3 {
                w += 0.2;
            } else if t_count == 2 {
                w += 0.1;
            }

            let pos = chunk.original_span.0 as f64;
            let pos_ratio = 1.0 - (pos / q_len);
            w += pos_ratio * 0.1;

            if chunk.intent == "relationship_query" {
                w += 0.05;
            } else if chunk.intent == "entity_lookup" {
                w += 0.1;
            }

            let avg_len = chunk.tokens.iter().map(|s| s.len()).sum::<usize>() as f64 / t_count.max(1) as f64;
            if avg_len > 6.0 {
                w += 0.1;
            }
            chunk.weight = w.min(1.0);
        }
    }

    fn deduplicate(&self, chunks: Vec<QueryChunk>) -> Vec<QueryChunk> {
        let mut seen = HashSet::new();
        let mut uniq = vec![];
        for c in chunks {
            let norm = c.text.to_lowercase().trim().to_string();
            if !seen.contains(&norm) {
                seen.insert(norm);
                uniq.push(c);
            }
        }
        uniq
    }
}
