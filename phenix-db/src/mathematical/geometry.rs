//! Geometry and vector transformation module (Khayyam, Tusi, Euler)

/// Manifold type for non-Euclidean geometry
#[derive(Debug, Clone, Copy)]
pub enum ManifoldType {
    /// Hyperbolic space
    Hyperbolic { curvature: f64 },
    /// Spherical geometry
    Spherical { radius: f64 },
    /// Euclidean space (fallback)
    Euclidean,
}

// TODO: Implement Cayley matrix transformations
// TODO: Implement curved space distance metrics
