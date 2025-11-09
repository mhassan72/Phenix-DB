//! Probability theory module (Kolmogorov, De Moivre)

/// Normalize probability distribution to sum to 1.0
pub fn normalize_probabilities(probabilities: &mut [f32]) {
    let sum: f32 = probabilities.iter().sum();
    if sum > 0.0 {
        for p in probabilities.iter_mut() {
            *p /= sum;
        }
    }
}

/// Verify probability distribution sums to 1.0 within tolerance
pub fn verify_probability_sum(probabilities: &[f32], tolerance: f32) -> bool {
    let sum: f32 = probabilities.iter().sum();
    (sum - 1.0).abs() < tolerance
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_probabilities() {
        let mut probs = vec![0.2, 0.3, 0.5];
        normalize_probabilities(&mut probs);
        assert!(verify_probability_sum(&probs, 0.001));
    }
}
