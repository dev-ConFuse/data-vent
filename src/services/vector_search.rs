use serde::{Deserialize, Serialize};
use serde_json::Value;
use tracing::info;

use crate::infra::Config;

#[allow(dead_code)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HNSWConfig {
    pub m: u32,
    pub ef_construction: u32,
    pub ef_runtime: u32,
    pub similarity_function: String,
    pub mode: String,
}

#[allow(dead_code)]
impl HNSWConfig {
    pub fn from_mode(mode: &str) -> Self {
        match mode.to_lowercase().as_str() {
            "high_recall" => Self {
                m: 32,
                ef_construction: 300,
                ef_runtime: 50,
                similarity_function: "COSINE".to_string(),
                mode: "high_recall".to_string(),
            },
            "low_latency" => Self {
                m: 16,
                ef_construction: 200,
                ef_runtime: 10,
                similarity_function: "COSINE".to_string(),
                mode: "low_latency".to_string(),
            },
            _ => Self {
                m: 24,
                ef_construction: 250,
                ef_runtime: 25,
                similarity_function: "COSINE".to_string(),
                mode: "balanced".to_string(),
            },
        }
    }

    pub fn from_config(config: &Config) -> Self {
        Self {
            m: config.hnsw_m,
            ef_construction: config.hnsw_ef_construction,
            ef_runtime: config.hnsw_ef_runtime,
            similarity_function: config.hnsw_similarity_function.clone(),
            mode: config.hnsw_mode.clone(),
        }
    }
}

#[derive(Clone)]
pub struct FalkorDBClient {
    _client: Option<redis::Client>,
    connection: Option<redis::aio::ConnectionManager>,
    is_dummy: bool,
}

impl FalkorDBClient {
    pub async fn new(config: &Config) -> anyhow::Result<Self> {
        let auth = if let Some(ref pwd) = config.falkordb_password {
            format!("{}:{}@", config.falkordb_username, pwd)
        } else {
            String::new()
        };
        
        let scheme = if config.falkordb_use_tls { "rediss" } else { "redis" };
        let url = format!("{}://{}{}:{}/{}", scheme, auth, config.falkordb_host, config.falkordb_port, config.falkordb_database);
        let client = redis::Client::open(url)?;
        
        // Bound the connection attempt so a slow FalkorDB cannot hang startup
        let connection = tokio::time::timeout(
            std::time::Duration::from_secs(30),
            redis::aio::ConnectionManager::new(client.clone())
        ).await
        .map_err(|_| anyhow::anyhow!("FalkorDB connection timed out after 30s"))??;
        
        Ok(Self {
            _client: Some(client),
            connection: Some(connection),
            is_dummy: false,
        })
    }

    pub fn new_dummy() -> Self {
        // Create a dummy client that will fail gracefully on queries
        // This allows the service to start even without FalkorDB
        Self {
            _client: None,
            connection: None,
            is_dummy: true,
        }
    }

    pub async fn initialize_indexes(&self, graph_name: &str, vector_dim: u16) -> anyhow::Result<()> {
        info!("Initializing FalkorDB indexes for graph: {} (dim: {})", graph_name, vector_dim);
        // Equivalent to db.idx.vector.createNodeIndex
        let cypher = format!("CALL db.idx.vector.createNodeIndex('Vector_Chunk', 'embeddings', {}, 'COSINE')", vector_dim);
        let _ = self.query(graph_name, &cypher).await;
        
        let fts_cypher = "CALL db.idx.fulltext.createNodeIndex('Vector_Chunk', 'content')";
        let _ = self.query(graph_name, fts_cypher).await;
        Ok(())
    }

    pub async fn update_hnsw_parameters(
        &self,
        graph_name: &str,
        config: &HNSWConfig,
    ) -> anyhow::Result<()> {
        let cypher = format!(
            "CALL db.idx.vector.configureNodeIndex('Vector_Chunk', 'embeddings', \
             {{M: {}, efConstruction: {}, efRuntime: {}, similarityFunction: '{}'}})",
            config.m, config.ef_construction, config.ef_runtime, config.similarity_function
        );
        self.query(graph_name, &cypher).await?;
        Ok(())
    }

    pub async fn query(&self, graph_name: &str, cypher: &str) -> anyhow::Result<Value> {
        if self.is_dummy || self.connection.is_none() {
            // Return empty result for dummy client
            return Ok(serde_json::json!({
                "results": [],
                "error": "FalkorDB not connected"
            }));
        }

        let conn = self.connection.as_ref().unwrap();
        let mut conn_clone = conn.clone();
        let result: redis::Value = redis::cmd("GRAPH.QUERY")
            .arg(graph_name)
            .arg(cypher)
            .arg("--compact")
            .query_async(&mut conn_clone)
            .await?;
            
        // We will return a rough JSON representation, but since parsing GRAPH.QUERY output is complex, 
        // we might need to write a custom parser in IntelligentRetriever.
        // For now we map it to string/Value loosely.
        let json_result = parse_redis_value_to_json(&result);
        Ok(json_result)
    }

    pub async fn _close(&self) -> anyhow::Result<()> {
        // Drop the multiplexed connection
        Ok(())
    }
}

// A helper to recursively turn a redis::Value into serde_json::Value
fn parse_redis_value_to_json(val: &redis::Value) -> Value {
    match val {
        redis::Value::Nil => Value::Null,
        redis::Value::Int(i) => Value::Number(serde_json::Number::from(*i)),
        redis::Value::Data(d) => {
            if let Ok(s) = std::str::from_utf8(d) {
                Value::String(s.to_string())
            } else {
                Value::String(format!("{:?}", d))
            }
        },
        redis::Value::Bulk(b) => {
            let vec: Vec<Value> = b.iter().map(parse_redis_value_to_json).collect();
            Value::Array(vec)
        },
        redis::Value::Status(s) => Value::String(s.clone()),
        redis::Value::Okay => Value::String("OK".to_string()),
    }
}
