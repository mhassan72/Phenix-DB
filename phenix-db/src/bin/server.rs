//! Phenix-DB memory substrate server

use phenix_db::{BUILD_INFO, VERSION};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing
    tracing_subscriber::fmt::init();

    tracing::info!("{}", BUILD_INFO);
    tracing::info!("Starting Phenix-DB server v{}", VERSION);

    // TODO: Initialize memory substrate components
    // TODO: Start gRPC and REST servers
    // TODO: Initialize distributed consciousness
    // TODO: Start observability endpoints

    tracing::info!("Phenix-DB server started successfully");

    // Keep server running
    tokio::signal::ctrl_c().await?;
    tracing::info!("Shutting down Phenix-DB server");

    Ok(())
}
