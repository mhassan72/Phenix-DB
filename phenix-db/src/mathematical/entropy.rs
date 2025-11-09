//! Entropy and information theory module (Shannon, Ibn al-Haytham)

/// Compute Shannon entropy for a probability distribution
pub fn shannon_entropy(probabilities: &[f32]) -> f64 {
    probabilities
        .iter()
        .filter(|&&p| p > 0.0)
        .map(|&p| {
            let p = p as f64;
            -p * p.log2()
        })
        .sum()
}

/// Normalize entropy to [0.0, 1.0] range
pub fn normalize_entropy(entropy: f64, alphabet_size: usize) -> f64 {
    if alphabet_size <= 1 {
        return 0.0;
    }
    entropy / (alphabet_size as f64).log2()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shannon_entropy() {
        // Uniform distribution has maximum entropy
        let uniform = vec![0.25, 0.25, 0.25, 0.25];
        let entropy = shannon_entropy(&uniform);
        assert!((entropy - 2.0).abs() < 0.0001); // log2(4) = 2
    }
}
