#![allow(dead_code)]
pub mod collection;
pub mod operators;

pub use self::collection::*;
use approx::AbsDiffEq;
use approx::RelativeEq;
use approx::UlpsEq;
use num_traits::Float;
use std::fmt::Display;
use std::iter::Sum;

use std::cmp::Ordering;

pub trait HFloat: Sum + Float + Display + AbsDiffEq + RelativeEq + UlpsEq {}
impl<T> HFloat for T
where
    T: Sum + Float + Display + AbsDiffEq + RelativeEq + UlpsEq,
{
}

#[derive(Abomonation, Debug, Clone, PartialEq)]
pub struct Histogram<T: Float> {
    bins: usize,
    data: Vec<Bin<T>>,
}

impl<T: HFloat> Histogram<T> {
    /// Initialize a new histogram maximum `bins` number of bins
    pub fn new(bins: usize) -> Histogram<T> {
        Histogram {
            bins,
            data: Vec::with_capacity(bins),
        }
    }

    /// Inserts a new data point into a bin, either creating a new bin to contain it
    /// or if the histogram already contains the maximum number of bins, merging it
    /// into an existing bin
    pub fn update(&mut self, p: T) {
        let bins = &mut self.data;

        match bins.binary_search_by(|probe| probe.p.partial_cmp(&p).unwrap_or(Ordering::Less)) {
            Ok(found_index) => bins[found_index].m = bins[found_index].m + flt(1.),
            Err(insert_at) => {
                bins.insert(insert_at, bin(p, flt(1.)));

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
    pub fn merge(&mut self, other: &Histogram<T>) {
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

        bins.sort_unstable();
        let mut i = 0;
        while i + 1 < bins.len() {
            if bins[i].p.ulps_eq(&bins[i + 1].p, T::default_epsilon(), 2) {
                let next_bin = bins[i + 1];
                bins[i].merge(&next_bin);
                bins.remove(i + 1);
            }
            i += 1;
        }
    }

    /// Estimates the number of points in the interval [-inf, b]
    pub fn sum(&self, b: T) -> T {
        if self.data.is_empty() {
            return T::zero();
        }

        if self.data[0].p > b {
            return T::zero();
        }

        // special case for only one bin; return 0 if b < p of that bin, m of the bin otherwise
        if self.data.len() == 1 {
            return self.data[0].p;
        }

        let i = (self.data
            .iter()
            .enumerate()
            .find(|(_, bin)| bin.p >= b)
            .unwrap_or_else(|| {
                let i = self.data.len() - 1;
                (i, &self.data[i])
            })
            .0)
            .max(1) - 1;

        let bin_i = self.data[i];
        let bin_i_next = self.data[i + 1];
        let two = flt(2.);
        let mut sum = {
            let m_b = bin_i.m + (bin_i_next.m - bin_i.m) / (bin_i_next.p - bin_i.p) * (b - bin_i.p);
            (bin_i.m + m_b) / two * (b - bin_i.p) / (bin_i_next.p - bin_i.p)
        };
        for j in 0..i {
            sum = sum + self.data[j].m;
        }

        debug_assert!(sum.is_finite());

        sum + bin_i.m / two
    }

    /// Estimates the number of points in the whole histogram
    pub fn sum_total(&self) -> T {
        self.data.iter().fold(flt(0.), |acc, bin| acc + bin.m)
    }

    #[allow(many_single_char_names)]
    pub fn uniform(&self, bins: usize) -> Option<Vec<T>> {
        if self.data.len() <= 1 {
            return None;
        }

        let m = |i: usize| self.data[i].m;
        let p = |i: usize| self.data[i].p;

        let uniform = (1..bins)
            .map(|j| {
                let s: T =
                    (flt::<T>(j as f64) / flt(bins as f64)) * self.data.iter().map(|b| b.m).sum::<T>();
                debug_assert!(self.data.len() > 1);
                let i = (1..self.data.len())
                    .find(|i| self.sum(p(*i)) > s)
                    .unwrap_or_else(|| self.data.len() - 1) - 1;

                let z = {
                    let d = s - self.sum(p(i));
                    let a = (m(i + 1) - m(i)).max(flt(1.));
                    let b = flt::<T>(2.) * m(i);
                    let c = flt::<T>(-2.) * d;
                    ((b * b - flt::<T>(4.) * a * c).sqrt() - b) / (flt::<T>(2.) * a)
                };
                p(i) + (p(i + 1) - p(i)) * z
            })
            .collect();
        Some(uniform)
    }

    pub fn candidate_splits(&self) -> Vec<T> {
        if self.data.len() > 1 {
            self.uniform(self.bins).unwrap()
        } else if self.data.len() == 1 {
            vec![self.data[0].p]
        } else {
            vec![]
        }
    }

    /// Returns a slice of the individual bins in this histogram
    pub fn bins(&self) -> &[Bin<T>] {
        self.data.as_slice()
    }
}

/// Initialize a Histogram from a `Vec<Bin>`, setting
/// the maximum number of bins to the number of bins in
/// the `Vec`
impl<T: Float> From<Vec<Bin<T>>> for Histogram<T> {
    fn from(bins: Vec<Bin<T>>) -> Histogram<T> {
        Histogram {
            bins: bins.len(),
            data: bins,
        }
    }
}

#[derive(Abomonation, Clone, Copy, Debug)]
pub struct Bin<T> {
    /// center value of the bin
    p: T,
    /// count/amount of items in the bin
    m: T,
}

impl<T: Float> Bin<T> {
    pub fn new(p: T, m: T) -> Bin<T> {
        Bin { p, m }
    }

    /// Merges this bin with another one, summing the number of points
    /// and shifting the center of the bin to accomodate
    pub fn merge(&mut self, other: &Bin<T>) {
        let m_sum = self.m + other.m;
        self.p = (self.p * self.m + other.p * other.m) / m_sum;
        self.m = m_sum;
    }
}

/// Sorts a bin by its center value (not by the number of points contained!)
impl<T: Float> PartialOrd for Bin<T> {
    fn partial_cmp(&self, other: &Bin<T>) -> Option<Ordering> {
        self.p.partial_cmp(&other.p)
    }
}

/// Compares the center and number of points in this bin with another.
/// Will fail in debug builds if any of the values are NaN or infinite
impl<T: Float> PartialEq for Bin<T> {
    fn eq(&self, other: &Bin<T>) -> bool {
        debug_assert!(self.p.is_finite());
        debug_assert!(other.p.is_finite());
        self.p == other.p && self.m == other.m
    }
}

impl<T: AbsDiffEq + Float> AbsDiffEq for Bin<T> where T::Epsilon: Copy {
    type Epsilon = T::Epsilon;

    fn default_epsilon() -> T::Epsilon {
        T::default_epsilon()
    }

    fn abs_diff_eq(&self, other: &Self, epsilon: T::Epsilon) -> bool {
        self.p.abs_diff_eq(&other.p, epsilon) && self.m.abs_diff_eq(&other.m, epsilon)
    }
}

impl<T: RelativeEq + Float> RelativeEq for Bin<T>
where
    T::Epsilon: Copy,
{
    fn default_max_relative() -> T::Epsilon {
        T::default_max_relative()
    }

    fn relative_eq(&self, other: &Self, epsilon: T::Epsilon, max_relative: T::Epsilon) -> bool {
        T::relative_eq(&self.p, &other.p, epsilon, max_relative)
            && T::relative_eq(&self.m, &other.m, epsilon, max_relative)
    }
}

impl<T: UlpsEq + Float> UlpsEq for Bin<T>
where
    T::Epsilon: Copy,
{
    fn default_max_ulps() -> u32 {
        T::default_max_ulps()
    }

    fn ulps_eq(&self, other: &Self, epsilon: T::Epsilon, max_ulps: u32) -> bool {
        T::ulps_eq(&self.m, &other.m, epsilon, max_ulps)
            && T::ulps_eq(&self.p, &other.p, epsilon, max_ulps)
    }
}

impl<T: HFloat> AbsDiffEq for Histogram<T> where T::Epsilon: Copy {
    type Epsilon = T::Epsilon;

    fn default_epsilon() -> T::Epsilon {
        T::default_epsilon()
    }

    fn abs_diff_eq(&self, other: &Self, epsilon: T::Epsilon) -> bool {
        self.bins()
            .iter()
            .zip(other.bins().iter())
            .all(|(a, b)| a.abs_diff_eq(b, epsilon)) && self.bins == other.bins
    }
}

impl<T: HFloat> UlpsEq for Histogram<T>
where
    T::Epsilon: Copy,
{
    fn default_max_ulps() -> u32 {
        T::default_max_ulps()
    }

    fn ulps_eq(&self, other: &Self, epsilon: T::Epsilon, max_ulps: u32) -> bool {
        self.bins()
            .iter()
            .zip(other.bins().iter())
            .all(|(a, b)| a.ulps_eq(b, epsilon, max_ulps)) && self.bins == other.bins
    }
}

impl<T: Float> Eq for Bin<T> {}

impl<T: Float> Ord for Bin<T> {
    fn cmp(&self, other: &Bin<T>) -> Ordering {
        self.p.partial_cmp(&other.p).unwrap()
    }
}

fn flt<T: Float>(primitive: f64) -> T {
    T::from(primitive).unwrap()
}

fn bin<T: Float>(p: T, m: T) -> Bin<T> {
    Bin::new(p, m)
}

#[cfg(test)]
mod test {
    use super::{bin, Histogram};

    const INPUT: &[f64] = &[23., 19., 10., 16., 36., 2., 9., 32., 30., 45.];

    #[test]
    fn update() {
        let mut hist: Histogram<f64> = Histogram::new(5);
        for i in &[23., 19., 10., 16., 36.] {
            hist.update(*i);
        }

        ulps_eq!(
            hist,
            &vec![
                bin(10., 1.),
                bin(16., 1.),
                bin(19., 1.),
                bin(23., 1.),
                bin(36., 1.),
            ].into(),
            max_ulps = 2,
        );

        hist.update(2.);
        ulps_eq!(
            hist,
            &vec![
                bin(2., 1.),
                bin(10., 1.),
                bin(17.5, 2.),
                bin(23., 1.),
                bin(36., 1.),
            ].into(),
            max_ulps = 2,
        );

        hist.update(9.);
        ulps_eq!(
            hist,
            &vec![
                bin(2., 1.),
                bin(9.5, 2.),
                bin(17.5, 2.),
                bin(23., 1.),
                bin(36., 1.),
            ].into(),
            max_ulps = 2,
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

        abs_diff_eq!(
            h1,
            &vec![
                bin(2., 1.),
                bin(9.5, 2.),
                bin(19.33333, 3.),
                bin(32.66666, 3.),
                bin(45., 1.),
            ].into(),
            epsilon = 0.00001,
        );
    }

    #[test]
    fn sum() {
        let h: Histogram<f64> = vec![
            bin(2., 1.),
            bin(9.5, 2.),
            bin(19.33, 3.),
            bin(32.67, 3.),
            bin(45., 1.),
        ].into();
        
        abs_diff_eq!(
            h.sum(15.),
            &3.275,
            epsilon = 0.001
        );
    }

    #[test]
    fn uniform() {
        let h: Histogram<f64> = vec![
            bin(2., 1.),
            bin(9.5, 2.),
            bin(19.33, 3.),
            bin(32.67, 3.),
            bin(45., 1.),
        ].into();
        println!("Uniform(3): {:?}", h.uniform(3));
    }
}
