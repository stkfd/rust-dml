use probability::distribution::{Gaussian, Inverse};
use std::cmp::Ordering;
use std::f64::INFINITY;
use std::f64::NEG_INFINITY;

#[derive(Debug, Clone)]
pub struct NormalQuantizer {
    ranges: Vec<(usize, f64, f64)>,
}

impl NormalQuantizer {
    pub fn new(mean: f64, std_dev: f64, steps: usize) -> Self {
        let dist = Gaussian::new(mean, std_dev);
        let steps: Vec<_> = (1..steps)
            .map(|s| dist.inverse(s as f64 / steps as f64))
            .collect();
        let mut ranges: Vec<_> = steps
            .iter()
            .cloned()
            .zip(steps.iter().cloned().skip(1))
            .enumerate()
            .map(|(idx, (lower, upper))| (idx + 1, lower, upper))
            .collect();

        ranges.push((0, NEG_INFINITY, steps[0]));
        ranges.push((steps.len(), steps[steps.len() - 1], INFINITY));

        // sort the ranges so that the ones closest to the mean come first
        // should improve performance during quantizing, assuming the samples
        // conform to the given distribution
        ranges.sort_unstable_by(|r1, r2| {
            (mean - (r1.1 + r1.2) / 2.)
                .abs()
                .partial_cmp(&(mean - (r2.1 + r2.2) / 2.).abs())
                .unwrap_or(Ordering::Less)
        });

        NormalQuantizer { ranges }
    }

    pub fn quantize(&self, num: f64) -> i64 {
        self.ranges
            .iter()
            .find(|(_idx, low, high)| num > *low && num <= *high)
            .unwrap()
            .0 as i64
    }
}

#[derive(Debug, Clone)]
pub struct UniformQuantizer {
    ranges: Vec<f64>,
}

impl UniformQuantizer {
    pub fn new(low: f64, high: f64, steps: usize) -> Self {
        let interval = high - low;
        let mut ranges: Vec<_> = (0..=steps)
            .map(|s| (s as f64 / steps as f64) * interval + low)
            .collect();
        ranges.insert(0, NEG_INFINITY);
        ranges.push(INFINITY);

        UniformQuantizer { ranges }
    }

    pub fn quantize(&self, num: f64) -> i64 {
        self.ranges
            .iter()
            .enumerate()
            .find(|(_idx, &threshold)| threshold > num)
            .unwrap()
            .0 as i64 - 1
    }
}
