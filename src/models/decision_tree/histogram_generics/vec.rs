use super::*;
use std::iter::FromIterator;
use vec_map::{Entry::*, Iter, VecMap};

#[derive(Clone, Debug)]
pub struct VecHistogramSet<H> {
    histograms: VecMap<H>,
}

#[derive(Abomonation, Clone)]
pub struct SerializableVecHistogramSet<H>(Vec<(usize, H)>);

impl<H> Default for VecHistogramSet<H> {
    fn default() -> Self {
        VecHistogramSet {
            histograms: VecMap::default(),
        }
    }
}

impl<H, Hs> From<VecHistogramSet<H>> for SerializableVecHistogramSet<Hs>
where
    H: HistogramSetItem,
    Hs: From<H>,
{
    /// Turn this item into a serializable version of itself
    fn from(set: VecHistogramSet<H>) -> Self {
        SerializableVecHistogramSet(
            set.histograms
                .into_iter()
                .map(|(k, v)| (k, Hs::from(v)))
                .collect(),
        )
    }
}

impl<H, Hs> Into<VecHistogramSet<H>> for SerializableVecHistogramSet<Hs>
where
    H: HistogramSetItem,
    Hs: Into<H>
{
    /// Recover a item from its serializable representation
    fn into(self) -> VecHistogramSet<H> {
        let histograms = self.0.into_iter().map(|(k, ser)| (k, ser.into()));
        VecHistogramSet {
            histograms: VecMap::from_iter(histograms),
        }
    }
}

impl<H> HistogramSetItem for VecHistogramSet<H>
where
    H: HistogramSetItem,
{
    type Serializable = SerializableVecHistogramSet<H::Serializable>;

    fn merge(&mut self, other: Self) {
        for (key, value) in other.histograms {
            match self.histograms.entry(key) {
                Occupied(mut entry) => {
                    entry.get_mut().merge(value);
                }
                Vacant(entry) => {
                    entry.insert(value);
                }
            }
        }
    }

    fn merge_borrowed(&mut self, other: &Self) {
        for (key, value) in other.histograms.iter() {
            match self.histograms.entry(key) {
                Occupied(mut entry) => {
                    entry.get_mut().merge_borrowed(value);
                }
                Vacant(entry) => {
                    entry.insert(value.clone());
                }
            }
        }
    }

    fn empty_clone(&self) -> Self {
        Self::default()
    }
}

impl<H> HistogramSet<usize, H> for VecHistogramSet<H>
where
    H: HistogramSetItem,
{
    fn get(&self, key: &usize) -> Option<&H> {
        self.histograms.get(*key)
    }

    fn get_mut(&mut self, key: &usize) -> Option<&mut H> {
        self.histograms.get_mut(*key)
    }

    fn get_or_insert_with(&mut self, key: &usize, insert_fn: impl Fn() -> H) -> &mut H {
        self.histograms.entry(key.clone()).or_insert_with(insert_fn)
    }
}

impl<H> VecHistogramSet<H> {
    pub fn iter(&self) -> Iter<H> {
        self.histograms.iter()
    }
}

impl<'a, H> IntoIterator for &'a VecHistogramSet<H> {
    type IntoIter = Iter<'a, H>;
    type Item = (usize, &'a H);

    fn into_iter(self) -> Iter<'a, H> {
        self.iter()
    }
}
