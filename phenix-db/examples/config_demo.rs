// Example demonstrating configuration loading and validation
//
// This example shows how to:
// - Load configuration from TOML file
// - Use default configuration
// - Validate configuration parameters
// - Access configuration values

use phenix_db::core::config::PhenixConfig;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Phenix-DB Configuration Demo ===\n");
    
    // 1. Create default configuration
    println!("1. Creating default configuration...");
    let default_config = PhenixConfig::default();
    println!("   ✓ Default configuration created");
    println!("   - Polynomial degree: {}", default_config.polynomial.degree);
    println!("   - PGM learning rate: {}", default_config.pgm.learning_rate);
    println!("   - Bellman discount factor: {}", default_config.bellman.discount_factor);
    println!("   - Compression target ratio: {}", default_config.compression.target_compression_ratio);
    println!("   - Learning convergence threshold: {}", default_config.learning.convergence_threshold);
    println!("   - Hot tier promotion threshold: {}", default_config.tiering.hot_promotion_threshold);
    println!("   - Distributed min replicas: {}", default_config.distributed.min_replicas);
    println!();
    
    // 2. Validate default configuration
    println!("2. Validating default configuration...");
    match default_config.validate() {
        Ok(_) => println!("   ✓ Default configuration is valid"),
        Err(e) => println!("   ✗ Validation error: {}", e),
    }
    println!();
    
    // 3. Try to load from file (if exists)
    println!("3. Attempting to load configuration from phenix-db.toml...");
    match PhenixConfig::from_file("phenix-db.toml") {
        Ok(config) => {
            println!("   ✓ Configuration loaded from file");
            println!("   - Polynomial degree: {}", config.polynomial.degree);
            println!("   - PGM learning rate: {}", config.pgm.learning_rate);
            println!("   - Bellman discount factor: {}", config.bellman.discount_factor);
        }
        Err(e) => {
            println!("   ℹ Could not load from file: {}", e);
            println!("   (This is expected if phenix-db.toml doesn't exist)");
        }
    }
    println!();
    
    // 4. Demonstrate validation errors
    println!("4. Demonstrating validation errors...");
    let mut invalid_config = PhenixConfig::default();
    
    // Invalid polynomial degree
    invalid_config.polynomial.degree = 0;
    match invalid_config.validate() {
        Ok(_) => println!("   ✗ Should have failed validation"),
        Err(e) => println!("   ✓ Caught validation error: {}", e),
    }
    
    // Invalid learning rate
    invalid_config.polynomial.degree = 5;
    invalid_config.pgm.learning_rate = 1.5;
    match invalid_config.validate() {
        Ok(_) => println!("   ✗ Should have failed validation"),
        Err(e) => println!("   ✓ Caught validation error: {}", e),
    }
    println!();
    
    // 5. Show mathematical parameters
    println!("5. Mathematical parameters summary:");
    let config = PhenixConfig::default();
    println!("   Polynomial Index (RPI):");
    println!("     - Degree: {} (Al-Karaji recursive evaluation)", config.polynomial.degree);
    println!("     - Precision: {} (error tolerance)", config.polynomial.precision_tolerance);
    println!();
    println!("   Probabilistic Graph Memory (PGM):");
    println!("     - Learning rate: {} (Kolmogorov probability)", config.pgm.learning_rate);
    println!("     - Pruning threshold: {} (edge removal)", config.pgm.pruning_threshold);
    println!("     - Co-access window: {}ms", config.pgm.co_access_window_ms);
    println!();
    println!("   Bellman Optimizer:");
    println!("     - Discount factor: {} (dynamic programming)", config.bellman.discount_factor);
    println!("     - Restructure threshold: {}x minimum cost", config.bellman.restructure_threshold);
    println!();
    println!("   Compression Engine (KCE):");
    println!("     - Target ratio: {} (Ramanujan series)", config.compression.target_compression_ratio);
    println!("     - Max decompression: {}ms", config.compression.max_decompression_time_ms);
    println!();
    println!("   Adaptive Learning:");
    println!("     - Learning rate: {} (PAC-learning)", config.learning.learning_rate);
    println!("     - Min accuracy: {} (prediction target)", config.learning.min_prediction_accuracy);
    println!();
    println!("   Memory Tiering:");
    println!("     - Hot promotion: {} accesses/hour", config.tiering.hot_promotion_threshold);
    println!("     - Warm promotion: {} accesses/hour", config.tiering.warm_promotion_threshold);
    println!();
    println!("   Distributed Consciousness:");
    println!("     - Global awareness: {}% (probabilistic sampling)", config.distributed.global_awareness_pct * 100.0);
    println!("     - Min replicas: {} (Von Neumann redundancy)", config.distributed.min_replicas);
    println!();
    
    println!("=== Configuration Demo Complete ===");
    
    Ok(())
}
