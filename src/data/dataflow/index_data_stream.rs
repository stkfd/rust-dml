use data::providers::IntSliceIndex;
use data::serialization::AbomonableArray2;
use data::serialization::AsView;
use ndarray::prelude::*;
use timely::dataflow::channels::pact::Pipeline;
use timely::dataflow::{operators::Operator, Scope, Stream};
use timely::Data;

pub trait IndexDataStream<S: Scope, T: Data> {
    fn index_data(&self) -> Stream<S, (IntSliceIndex<usize>, AbomonableArray2<T>)>;
}

impl<S, T> IndexDataStream<S, T> for Stream<S, AbomonableArray2<T>>
where
    S: Scope,
    T: Data,
{
    fn index_data(&self) -> Stream<S, (IntSliceIndex<usize>, AbomonableArray2<T>)> {
        self.unary(Pipeline, "IndexData", |_, _| {
            let mut count = 0;
            move |input, output| {
                input.for_each(|time, data| {
                    let mut session = output.session(&time);
                    for array in data.drain(..) {
                        let start = count;
                        let length = array.view().len_of(Axis(0));
                        count = count + length;
                        session.give((IntSliceIndex { start, length }, array));
                    }
                });
            }
        })
    }
}
