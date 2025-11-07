//! # Phenix DB Server Binary
//!
//! Main server binary for running Phenix DB in production environments.

use phenix_db::api::PhenixDB;
use std::env;
use tracing::{info, error};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing
    tracing_subscriber::fmt::init();

    info!("Starting Phenix DB Server v{}", phenix_db::VERSION);

    // Parse command line arguments
    let args: Vec<String> = env::args().collect();
    let config_path = args.get(1).map(|s| s.as_str()).unwrap_or("phenix.toml");

    // Initialize database
    let db = match PhenixDB::builder()
        .with_config_file(config_path)
        .build()
        .await
    {
        Ok(db) => {
            info!("Phenix DB initialized successfully");
            db
        }
        Err(e) => {
            error!("Failed to initialize Phenix DB: {}", e);
            return Err(e.into());
        }
    };

    info!("Phenix DB server started successfully");
    
    // TODO: Start gRPC and REST servers in later tasks
    // For now, just keep the server running
    tokio::signal::ctrl_c().await?;
    
    info!("Shutting down Phenix DB server");
    Ok(())
}