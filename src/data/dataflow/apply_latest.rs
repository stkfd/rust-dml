use fnv::FnvHashMap;
use timely::dataflow::channels::pact::Pipeline;
use timely::dataflow::{operators::Operator, Scope, Stream};
use timely::Data;

/// Extension trait for `Stream`.
pub trait ApplyLatest<S: Scope, D: Data> {
    /// Applies the latest item in this stream to incoming
    /// items in the other passed stream
    fn apply_latest<D2: Data, D3: Data>(
        &self,
        inputs: &Stream<S, D2>,
        callback: impl Fn(&S::Timestamp, &D, D2) -> D3 + 'static,
    ) -> Stream<S, D3>;
}

impl<S: Scope, D: Data> ApplyLatest<S, D> for Stream<S, D> {
    fn apply_latest<D2: Data, D3: Data>(
        &self,
        inputs: &Stream<S, D2>,
        callback: impl Fn(&S::Timestamp, &D, D2) -> D3 + 'static,
    ) -> Stream<S, D3> {
        self.binary(inputs, Pipeline, Pipeline, "Predict", |_, _| {
            let mut latest_item_opt = None;
            let mut input_stash = FnvHashMap::default();

            move |trees, inputs, output| {
                trees.for_each(|_, data| {
                    latest_item_opt = Some(data.drain(..).last().expect("Latest decision tree"));
                });
                inputs.for_each(|time, ref mut data| {
                    if let Some(latest_item) = &latest_item_opt {
                        output.session(&time).give_iterator(
                            data.drain(..)
                                .map(|datum| callback(&time, latest_item, datum)),
                        );
                    } else {
                        input_stash
                            .entry(time.retain())
                            .or_insert_with(Vec::new)
                            .extend(data.drain(..));
                    }
                });

                if let Some(latest_item) = &latest_item_opt {
                    for (ref time, ref mut data) in &mut input_stash {
                        output.session(&time).give_iterator(
                            data.drain(..)
                                .map(|datum| callback(&time, latest_item, datum)),
                        );
                        //data.clear();
                    }
                }
                input_stash.retain(|_, data| !data.is_empty());
            }
        })
    }
}
