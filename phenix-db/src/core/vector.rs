//! Vector operations and data structures

use serde::{Deserialize, Serialize};

/// High-dimensional vector embedding
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Vector {
    /// Number of dimensions
    pub dimensions: usize,
    
    /// Vector values
    pub values: Vec<f32>,
    
    /// Precomputed L2 norm for efficiency
    pub norm: f32,
}

impl Vector {
    /// Create a new vector from values
    pub fn new(values: Vec<f32>) -> Self {
        let dimensions = values.len();
        let norm = Self::compute_norm(&values);
        Self {
            dimensions,
            values,
            norm,
        }
    }

    /// Compute L2 norm of a vector
    fn compute_norm(values: &[f32]) -> f32 {
        values.iter().map(|v| v * v).sum::<f32>().sqrt()
    }

    /// Recompute and update the norm
    pub fn update_norm(&mut self) {
        self.norm = Self::compute_norm(&self.values);
    }

    /// Compute cosine similarity with another vector
    pub fn cosine_similarity(&self, other: &Vector) -> f32 {
        if self.dimensions != other.dimensions {
            return 0.0;
        }

        let dot_product: f32 = self
            .values
            .iter()
            .zip(other.values.iter())
            .map(|(a, b)| a * b)
            .sum();

        dot_product / (self.norm * other.norm)
    }

    /// Compute Euclidean distance to another vector
    pub fn euclidean_distance(&self, other: &Vector) -> f32 {
        if self.dimensions != other.dimensions {
            return f32::INFINITY;
        }

        self.values
            .iter()
            .zip(other.values.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vector_creation() {
        let values = vec![1.0, 2.0, 3.0];
        let vector = Vector::new(values.clone());
        
        assert_eq!(vector.dimensions, 3);
        assert_eq!(vector.values, values);
        assert!((vector.norm - 3.7416575).abs() < 0.0001);
    }

    #[test]
    fn test_cosine_similarity() {
        let v1 = Vector::new(vec![1.0, 0.0, 0.0]);
        let v2 = Vector::new(vec![1.0, 0.0, 0.0]);
        let v3 = Vector::new(vec![0.0, 1.0, 0.0]);
        
        assert!((v1.cosine_similarity(&v2) - 1.0).abs() < 0.0001);
        assert!((v1.cosine_similarity(&v3) - 0.0).abs() < 0.0001);
    }
}
