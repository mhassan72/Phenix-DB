//! # Vector Operations and Data Structures
//!
//! This module implements vector data structures and operations for Phenix DB.
//! Vectors are dense float32 arrays representing embeddings that can be used
//! for similarity search and machine learning applications.

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::core::error::{PhenixDBError, Result};
use crate::defaults::{MIN_VECTOR_DIMENSIONS, MAX_VECTOR_DIMENSIONS};

/// Unique identifier for vectors in the database
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct VectorId(pub Uuid);

impl VectorId {
    /// Generate a new random VectorId
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }

    /// Create VectorId from string representation
    pub fn from_string(s: &str) -> Result<Self> {
        let uuid = Uuid::parse_str(s)
            .map_err(|e| PhenixDBError::InvalidVectorId(format!("Invalid UUID format: {}", e)))?;
        Ok(Self(uuid))
    }

    /// Get string representation of VectorId
    pub fn to_string(&self) -> String {
        self.0.to_string()
    }
}

impl Default for VectorId {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for VectorId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Encryption algorithm used for vector encryption
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EncryptionAlgorithm {
    /// AES-GCM encryption
    AesGcm,
    /// ChaCha20-Poly1305 encryption
    ChaCha20Poly1305,
}

impl std::fmt::Display for EncryptionAlgorithm {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EncryptionAlgorithm::AesGcm => write!(f, "AES-GCM"),
            EncryptionAlgorithm::ChaCha20Poly1305 => write!(f, "ChaCha20-Poly1305"),
        }
    }
}

/// Vector component of Entity containing dense float32 embeddings
///
/// The Vector represents dense embeddings that can be used for similarity search,
/// machine learning, and AI applications. Vectors support normalization,
/// compression, and encryption for optimal performance and security.
///
/// # Design Principles
/// - **Configurable Dimensions**: Support for 128-4096 dimensions
/// - **Performance Optimized**: Efficient memory layout and SIMD operations
/// - **Security First**: Built-in encryption support with multiple algorithms
/// - **Compression Ready**: Metadata for compression ratios and algorithms
///
/// # Examples
/// ```rust
/// use phenix_db::core::vector::Vector;
///
/// // Create a simple vector
/// let vector = Vector::new(vec![0.1, 0.2, 0.3, 0.4]);
///
/// // Create a normalized vector
/// let mut vector = Vector::new(vec![1.0, 2.0, 3.0]);
/// vector.normalize();
///
/// // Check vector properties
/// assert_eq!(vector.dimension_count(), 3);
/// assert!(vector.is_normalized());
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Vector {
    /// Dense float32 array representing the embedding
    pub dimensions: Vec<f32>,
    
    /// Number of dimensions (cached for performance)
    pub dimension_count: usize,
    
    /// L2 norm of the vector (cached after normalization)
    pub norm: f32,
    
    /// Compression ratio if vector is compressed (None if uncompressed)
    pub compression_ratio: Option<f32>,
    
    /// Encryption algorithm used (None if unencrypted)
    pub encryption_algorithm: Option<EncryptionAlgorithm>,
    
    /// Whether the vector has been normalized
    pub is_normalized: bool,
}

impl Vector {
    /// Create a new Vector from dimensions
    pub fn new(dimensions: Vec<f32>) -> Self {
        let dimension_count = dimensions.len();
        let norm = Self::calculate_norm(&dimensions);
        
        Self {
            dimensions,
            dimension_count,
            norm,
            compression_ratio: None,
            encryption_algorithm: None,
            is_normalized: false,
        }
    }

    /// Create a normalized vector from dimensions
    pub fn new_normalized(mut dimensions: Vec<f32>) -> Self {
        let dimension_count = dimensions.len();
        Self::normalize_in_place(&mut dimensions);
        
        Self {
            dimensions,
            dimension_count,
            norm: 1.0,
            compression_ratio: None,
            encryption_algorithm: None,
            is_normalized: true,
        }
    }

    /// Get the number of dimensions
    pub fn dimension_count(&self) -> usize {
        self.dimension_count
    }

    /// Get the L2 norm of the vector
    pub fn norm(&self) -> f32 {
        self.norm
    }

    /// Check if the vector is normalized
    pub fn is_normalized(&self) -> bool {
        self.is_normalized
    }

    /// Normalize the vector in-place
    pub fn normalize(&mut self) {
        if !self.is_normalized {
            Self::normalize_in_place(&mut self.dimensions);
            self.norm = 1.0;
            self.is_normalized = true;
        }
    }

    /// Get a normalized copy of the vector
    pub fn normalized(&self) -> Self {
        if self.is_normalized {
            self.clone()
        } else {
            Self::new_normalized(self.dimensions.clone())
        }
    }

    /// Calculate cosine similarity with another vector
    pub fn cosine_similarity(&self, other: &Vector) -> Result<f32> {
        if self.dimension_count != other.dimension_count {
            return Err(PhenixDBError::DimensionMismatch {
                expected: self.dimension_count,
                actual: other.dimension_count,
            });
        }

        let dot_product = self.dot_product(other);
        
        if self.is_normalized && other.is_normalized {
            Ok(dot_product)
        } else {
            let magnitude = self.norm * other.norm;
            if magnitude == 0.0 {
                Ok(0.0)
            } else {
                Ok(dot_product / magnitude)
            }
        }
    }

    /// Calculate dot product with another vector
    pub fn dot_product(&self, other: &Vector) -> f32 {
        self.dimensions
            .iter()
            .zip(other.dimensions.iter())
            .map(|(a, b)| a * b)
            .sum()
    }

    /// Calculate Euclidean distance to another vector
    pub fn euclidean_distance(&self, other: &Vector) -> Result<f32> {
        if self.dimension_count != other.dimension_count {
            return Err(PhenixDBError::DimensionMismatch {
                expected: self.dimension_count,
                actual: other.dimension_count,
            });
        }

        let sum_squared_diff: f32 = self.dimensions
            .iter()
            .zip(other.dimensions.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum();

        Ok(sum_squared_diff.sqrt())
    }

    /// Calculate Manhattan distance to another vector
    pub fn manhattan_distance(&self, other: &Vector) -> Result<f32> {
        if self.dimension_count != other.dimension_count {
            return Err(PhenixDBError::DimensionMismatch {
                expected: self.dimension_count,
                actual: other.dimension_count,
            });
        }

        let sum_abs_diff: f32 = self.dimensions
            .iter()
            .zip(other.dimensions.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();

        Ok(sum_abs_diff)
    }

    /// Validate vector constraints
    pub fn validate(&self) -> Result<()> {
        // Check dimension count
        if self.dimension_count < MIN_VECTOR_DIMENSIONS {
            return Err(PhenixDBError::ValidationError {
                message: format!("Vector dimension count {} is below minimum {}", 
                       self.dimension_count, MIN_VECTOR_DIMENSIONS)
            });
        }

        if self.dimension_count > MAX_VECTOR_DIMENSIONS {
            return Err(PhenixDBError::ValidationError {
                message: format!("Vector dimension count {} exceeds maximum {}", 
                       self.dimension_count, MAX_VECTOR_DIMENSIONS)
            });
        }

        // Check that dimensions array matches dimension count
        if self.dimensions.len() != self.dimension_count {
            return Err(PhenixDBError::ValidationError {
                message: format!("Dimension array length {} does not match dimension count {}", 
                       self.dimensions.len(), self.dimension_count)
            });
        }

        // Check for invalid float values
        for (i, &dim) in self.dimensions.iter().enumerate() {
            if !dim.is_finite() {
                return Err(PhenixDBError::ValidationError {
                    message: format!("Invalid float value at dimension {}: {}", i, dim)
                });
            }
        }

        // Validate norm calculation
        let calculated_norm = Self::calculate_norm(&self.dimensions);
        let norm_diff = (self.norm - calculated_norm).abs();
        if norm_diff > 1e-6 {
            return Err(PhenixDBError::ValidationError {
                message: format!("Cached norm {} does not match calculated norm {}", 
                       self.norm, calculated_norm)
            });
        }

        Ok(())
    }

    /// Calculate approximate memory usage
    pub fn memory_usage(&self) -> usize {
        std::mem::size_of::<Self>() + (self.dimensions.len() * std::mem::size_of::<f32>())
    }

    /// Set compression metadata
    pub fn set_compression_ratio(&mut self, ratio: f32) {
        self.compression_ratio = Some(ratio);
    }

    /// Set encryption metadata
    pub fn set_encryption_algorithm(&mut self, algorithm: EncryptionAlgorithm) {
        self.encryption_algorithm = Some(algorithm);
    }

    /// Check if vector is compressed
    pub fn is_compressed(&self) -> bool {
        self.compression_ratio.is_some()
    }

    /// Check if vector is encrypted
    pub fn is_encrypted(&self) -> bool {
        self.encryption_algorithm.is_some()
    }

    // Private helper methods

    /// Calculate L2 norm of dimensions
    fn calculate_norm(dimensions: &[f32]) -> f32 {
        dimensions.iter().map(|x| x * x).sum::<f32>().sqrt()
    }

    /// Normalize dimensions in-place
    fn normalize_in_place(dimensions: &mut [f32]) {
        let norm = Self::calculate_norm(dimensions);
        if norm > 0.0 {
            for dim in dimensions.iter_mut() {
                *dim /= norm;
            }
        }
    }
}

impl PartialEq for Vector {
    fn eq(&self, other: &Self) -> bool {
        self.dimension_count == other.dimension_count &&
        self.dimensions.len() == other.dimensions.len() &&
        self.dimensions.iter().zip(other.dimensions.iter()).all(|(a, b)| (a - b).abs() < f32::EPSILON)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vector_creation() {
        let dimensions = vec![1.0, 2.0, 3.0];
        let vector = Vector::new(dimensions.clone());
        
        assert_eq!(vector.dimensions, dimensions);
        assert_eq!(vector.dimension_count(), 3);
        assert!(!vector.is_normalized());
    }

    #[test]
    fn test_vector_normalization() {
        let mut vector = Vector::new(vec![3.0, 4.0]);
        assert!(!vector.is_normalized());
        
        vector.normalize();
        assert!(vector.is_normalized());
        assert!((vector.norm() - 1.0).abs() < f32::EPSILON);
        
        // Check normalized values
        assert!((vector.dimensions[0] - 0.6).abs() < 1e-6);
        assert!((vector.dimensions[1] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity() {
        let v1 = Vector::new_normalized(vec![1.0, 0.0]);
        let v2 = Vector::new_normalized(vec![0.0, 1.0]);
        let v3 = Vector::new_normalized(vec![1.0, 0.0]);
        
        assert!((v1.cosine_similarity(&v2).unwrap() - 0.0).abs() < f32::EPSILON);
        assert!((v1.cosine_similarity(&v3).unwrap() - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_euclidean_distance() {
        let v1 = Vector::new(vec![0.0, 0.0]);
        let v2 = Vector::new(vec![3.0, 4.0]);
        
        let distance = v1.euclidean_distance(&v2).unwrap();
        assert!((distance - 5.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_dimension_mismatch() {
        let v1 = Vector::new(vec![1.0, 2.0]);
        let v2 = Vector::new(vec![1.0, 2.0, 3.0]);
        
        assert!(v1.cosine_similarity(&v2).is_err());
        assert!(v1.euclidean_distance(&v2).is_err());
    }

    #[test]
    fn test_vector_validation() {
        let valid_vector = Vector::new(vec![1.0; 256]);
        assert!(valid_vector.validate().is_ok());
        
        let invalid_vector = Vector::new(vec![1.0; 50]); // Below minimum
        assert!(invalid_vector.validate().is_err());
    }

    #[test]
    fn test_vector_metadata() {
        let mut vector = Vector::new(vec![1.0, 2.0, 3.0]);
        
        assert!(!vector.is_compressed());
        assert!(!vector.is_encrypted());
        
        vector.set_compression_ratio(0.7);
        vector.set_encryption_algorithm(EncryptionAlgorithm::AesGcm);
        
        assert!(vector.is_compressed());
        assert!(vector.is_encrypted());
        assert_eq!(vector.compression_ratio, Some(0.7));
        assert_eq!(vector.encryption_algorithm, Some(EncryptionAlgorithm::AesGcm));
    }
}