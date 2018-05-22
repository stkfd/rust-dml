#![allow(dead_code)]
pub mod operators;

use float_cmp::{ApproxEq, Ulps};
use std::cmp::Ordering;

#[derive(Abomonation, Debug, Clone)]
pub struct Histogram {
    bins: usize,
    data: Vec<Bin>,
}

impl Histogram {
    /// Initialize a new histogram maximum `bins` number of bins
    pub fn new(bins: usize) -> Histogram {
        Histogram {
            bins,
            data: Vec::with_capacity(bins),
        }
    }

    /// Inserts a new data point into a bin, either creating a new bin to contain it
    /// or if the histogram already contains the maximum number of bins, merging it
    /// into an existing bin
    pub fn update(&mut self, p: f64) {
        let bins = &mut self.data;

        match bins.binary_search_by(|probe| probe.p.partial_cmp(&p).unwrap_or(Ordering::Less)) {
            Ok(found_index) => bins[found_index].m += 1.,
            Err(insert_at) => {
                bins.insert(insert_at, bin(p, 1.));

                if bins.len() > self.bins {
                    // find index of the two closest together bins
                    let least_diff = bins.iter()
                        .zip(bins.iter().skip(1))
                        .map(|(current, next)| next.p - current.p)
                        .enumerate()
                        .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                        .unwrap()
                        .0;

                    let next_bin = bins[least_diff + 1];
                    bins[least_diff].merge(&next_bin);
                    bins.remove(least_diff + 1);
                }
            }
        }
    }

    /// Merges the contents of another histogram into this histogram
    pub fn merge(&mut self, other: &Histogram) {
        let bins = &mut self.data;
        let other_bins = &other.data;
        bins.extend(other_bins);

        while bins.len() > self.bins {
            let least_diff = bins.iter()
                .zip(bins.iter().skip(1))
                .map(|(current, next)| next.p - current.p)
                .enumerate()
                .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .unwrap()
                .0;

            let next_bin = bins[least_diff + 1];
            bins[least_diff].merge(&next_bin);
            bins.remove(least_diff + 1);
        }
    }

    /// Estimates the number of points in the interval [-inf, b]
    pub fn sum(&self, b: f64) -> f64 {
        debug!("Sum for {:?} b = {}", self, b);
        let i = (self.data
            .iter()
            .enumerate()
            .find(|(_, bin)| bin.p >= b)
            .unwrap_or_else(|| { let i = self.data.len() - 1; (i, &self.data[i]) })
            .0)
            .max(1) - 1;

        let bin_i = self.data[i];
        let bin_i_next = self.data[i + 1];
        let mut sum = {
            let m_b = bin_i.m + (bin_i_next.m - bin_i.m) / (bin_i_next.p - bin_i.p) * (b - bin_i.p);
            (bin_i.m + m_b) / 2. * (b - bin_i.p) / (bin_i_next.p - bin_i.p)
        };
        for j in 0..i {
            sum += self.data[j].m;
        }
        sum + bin_i.m / 2.
    }

    /// Estimates the number of points in the whole histogram
    pub fn sum_total(&self) -> f64 {
        self.data.iter().fold(0., |acc, bin| acc + bin.m)
    }

    #[allow(many_single_char_names)]
    pub fn uniform(&self, bins: usize) -> Vec<f64> {
        let m = |i: usize| self.data[i].m;
        let p = |i: usize| self.data[i].p;

        (1..bins)
            .map(|j| {
                let s = (j as f64 / bins as f64) * self.data.iter().map(|b| b.m).sum::<f64>();
                debug!("s = {}", s);
                let i = (1..self.data.len())
                    .find(|i| {
                        let sum = self.sum(p(*i));
                        debug!("Sum = {}, s = {}", sum, s);
                        sum > s
                    })
                    .unwrap_or_else(|| self.data.len() - 1) - 1;

                let z = {
                    let d = s - self.sum(p(i));
                    let a = (m(i + 1) - m(i)).max(1.);
                    let b = 2. * m(i);
                    let c = -2. * d;
                    ((b * b - 4. * a * c).sqrt() - b) / (2. * a)
                };
                p(i) + (p(i + 1) - p(i)) * z
            })
            .collect()
    }

    /// Returns a slice of the individual bins in this histogram
    pub fn bins(&self) -> &[Bin] {
        self.data.as_slice()
    }
}

/// Initialize a Histogram from a `Vec<Bin>`, setting
/// the maximum number of bins to the number of bins in
/// the `Vec`
impl From<Vec<Bin>> for Histogram {
    fn from(bins: Vec<Bin>) -> Histogram {
        Histogram {
            bins: bins.len(),
            data: bins,
        }
    }
}

#[derive(Abomonation, Clone, Copy, Debug)]
pub struct Bin {
    /// center value of the bin
    p: f64,
    /// count/amount of items in the bin
    m: f64,
}

impl Bin {
    pub fn new(p: f64, m: f64) -> Bin {
        Bin { p, m }
    }

    /// Merges this bin with another one, summing the number of points
    /// and shifting the center of the bin to accomodate
    pub fn merge(&mut self, other: &Bin) {
        let m_sum = self.m + other.m;
        self.p = (self.p * self.m + other.p * other.m) / m_sum;
        self.m = m_sum;
    }
}

/// Sorts a bin by its center value (not by the number of points contained!)
impl PartialOrd for Bin {
    fn partial_cmp(&self, other: &Bin) -> Option<Ordering> {
        self.p.partial_cmp(&other.p)
    }
}

/// Compares the center and number of points in this bin with another.
/// Will fail in debug builds if any of the values are NaN or infinite
impl PartialEq for Bin {
    fn eq(&self, other: &Bin) -> bool {
        debug_assert!(self.p.is_finite());
        debug_assert!(other.p.is_finite());
        self.p == other.p && self.m == other.m
    }
}

impl ApproxEq for Bin {
    type Flt = f64;

    fn approx_eq(&self, other: &Self, epsilon: Self::Flt, ulps: <Self::Flt as Ulps>::U) -> bool {
        self.p.approx_eq(&other.p, epsilon, ulps) && self.m.approx_eq(&other.m, epsilon, ulps)
    }
}

impl ApproxEq for Histogram {
    type Flt = f64;

    fn approx_eq(&self, other: &Self, epsilon: Self::Flt, ulps: <Self::Flt as Ulps>::U) -> bool {
        self.bins()
            .iter()
            .zip(other.bins().iter())
            .all(|(a, b)| a.approx_eq(b, epsilon, ulps)) && self.bins == other.bins
    }
}

impl Eq for Bin {}

impl Ord for Bin {
    fn cmp(&self, other: &Bin) -> Ordering {
        self.p.partial_cmp(&other.p).unwrap()
    }
}

fn bin(p: f64, m: f64) -> Bin {
    Bin::new(p, m)
}

#[cfg(test)]
mod test {
    use super::*;
    use float_cmp::ApproxEq;

    const INPUT: &[f64] = &[23., 19., 10., 16., 36., 2., 9., 32., 30., 45.];

    #[test]
    fn update() {
        let mut hist = Histogram::new(5);
        for i in &[23., 19., 10., 16., 36.] {
            hist.update(*i);
        }

        assert!(
            hist.approx_eq(
                &vec![
                    bin(10., 1.),
                    bin(16., 1.),
                    bin(19., 1.),
                    bin(23., 1.),
                    bin(36., 1.),
                ].into(),
                ::std::f64::EPSILON,
                2,
            )
        );

        hist.update(2.);
        assert!(
            hist.approx_eq(
                &vec![
                    bin(2., 1.),
                    bin(10., 1.),
                    bin(17.5, 2.),
                    bin(23., 1.),
                    bin(36., 1.),
                ].into(),
                ::std::f64::EPSILON,
                2,
            )
        );

        hist.update(9.);
        assert!(
            hist.approx_eq(
                &vec![
                    bin(2., 1.),
                    bin(9.5, 2.),
                    bin(17.5, 2.),
                    bin(23., 1.),
                    bin(36., 1.),
                ].into(),
                ::std::f64::EPSILON,
                2,
            )
        );
    }

    #[test]
    fn merge() {
        let mut h1 = [23., 19., 10., 16., 36., 2., 9.].iter().fold(
            Histogram::new(5),
            |mut h, i| {
                h.update(*i);
                h
            },
        );
        let h2 = [32., 30., 45.].iter().fold(Histogram::new(5), |mut h, i| {
            h.update(*i);
            h
        });
        h1.merge(&h2);

        assert!(
            h1.approx_eq(
                &vec![
                    bin(2., 1.),
                    bin(9.5, 2.),
                    bin(19.33333, 3.),
                    bin(32.66666, 3.),
                    bin(45., 1.),
                ].into(),
                0.00001,
                3,
            )
        );
    }

    #[test]
    fn sum() {
        let h: Histogram = vec![
            bin(2., 1.),
            bin(9.5, 2.),
            bin(19.33, 3.),
            bin(32.67, 3.),
            bin(45., 1.),
        ].into();
        assert!(h.sum(15.).approx_eq(&3.275, 0.001, 2));
    }

    #[test]
    fn uniform() {
        let h: Histogram = vec![
            bin(2., 1.),
            bin(9.5, 2.),
            bin(19.33, 3.),
            bin(32.67, 3.),
            bin(45., 1.),
        ].into();
        println!("Uniform(3): {:?}", h.uniform(3));
    }
}
