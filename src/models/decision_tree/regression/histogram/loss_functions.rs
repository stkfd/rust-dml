use num_traits::Float;
use models::decision_tree::histogram_generics::ContinuousValue;
use models::decision_tree::regression::histogram::{Histogram, PartialBinSum};
use models::decision_tree::histogram_generics::BaseHistogram;

pub trait TrimmedLad<L> {
    fn trimmed_lad(&self, trim_ratio: L) -> L;
}

impl<L: ContinuousValue> TrimmedLad<L> for Histogram<L> {
    fn trimmed_lad(&self, trim_ratio: L) -> L {
        let count_total = self.count();
        let count_total_f = L::from(count_total).unwrap();
        
        if count_total == 0 { return L::zero() }

        // calculate sample count thresholds for the relevant quantiles at 0.5, trim_ratio and 1 - trim_ratio
        let half_q_threshold = (count_total as f64 * 0.5).round() as u64;
        let lower_trim_q_threshold = (L::from(count_total).unwrap() * trim_ratio)
            .round()
            .to_u64()
            .unwrap();
        let upper_trim_q_threshold = count_total - lower_trim_q_threshold;

        let mut half_q = None;
        let mut lower_trim_q = None;
        let mut upper_trim_q = None;

        let mut sample_count = 0;
        let mut s = L::zero();
        let mut s_t = L::zero();

        for (bin_addr, bin_data) in self.bins() {
            sample_count += bin_data.count;

            if half_q.is_none() && sample_count >= half_q_threshold {
                half_q = Some(((bin_addr, bin_data), sample_count - bin_data.count));
            }

            if lower_trim_q.is_none() && sample_count >= lower_trim_q_threshold {
                lower_trim_q = Some(((bin_addr, bin_data), sample_count - bin_data.count));
            }

            if upper_trim_q.is_none() && sample_count >= upper_trim_q_threshold {
                upper_trim_q = Some(((bin_addr, bin_data), sample_count - bin_data.count));
            }

            match half_q {
                Some(_) => s = s + bin_data.sum,
                None => s = s - bin_data.sum,
            }

            match (lower_trim_q, half_q) {
                // lower quantile found, half quantile not found
                (Some(_), None) => s_t = s_t - bin_data.sum,
                // upper quantile not found, half quantile found
                (None, Some(_)) => s_t = s_t + bin_data.sum,
                _ => {}
            }
        }

        // unwrap quantile bins; after the loop, these should always have a Some(_) value
        let (half_q_bin, half_q_sum) = half_q.unwrap();
        let (lower_q_bin, lower_q_sum) = lower_trim_q.unwrap();
        let (upper_q_bin, upper_q_sum) = upper_trim_q.unwrap();

        let r = count_total / 2 - half_q_sum;
        let r1 = (count_total_f * trim_ratio).to_u64().unwrap() - lower_q_sum;
        let r2 = (count_total_f * (L::one() - trim_ratio)).to_u64().unwrap() - upper_q_sum;

        s_t + lower_q_bin.partial_sum(r1) - lower_q_bin.1.sum
            + upper_q_bin.partial_sum(r2)
            + half_q_bin.1.sum - L::from(2.).unwrap() * half_q_bin.partial_sum(r)
    }
}

pub trait WeightedLoss<L: Float> {
    fn weighted_loss(&self, h_total: &Histogram<L>, h_left: &Histogram<L>, h_right: &Histogram<L>) -> L;
}

#[derive(Clone, Copy, Constructor)]
pub struct TrimmedLadWeightedLoss<L>(pub L);

impl<L: ContinuousValue> WeightedLoss<L> for TrimmedLadWeightedLoss<L> {
    fn weighted_loss(&self, h_total: &Histogram<L>, h_left: &Histogram<L>, h_right: &Histogram<L>) -> L {
        let lad_l = h_left.trimmed_lad(self.0);
        let count_l = L::from(h_left.count()).unwrap();

        let lad_r = h_right.trimmed_lad(self.0);
        let count_r = L::from(h_right.count()).unwrap();

        let count_n = L::from(h_total.count()).unwrap();

        lad_l * (count_l / count_n) + lad_r * (count_r / count_n)
    }
}
