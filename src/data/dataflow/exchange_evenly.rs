use timely::dataflow::channels::pact::Pipeline;
use timely::dataflow::operators::*;
use timely::dataflow::{Scope, Stream};
use timely::ExchangeData;

/// Extension trait for `Stream`.
pub trait ExchangeEvenly<S: Scope, D: ExchangeData> {
    /// Exchanges records so they are evenly distributed between
    /// all available workers.
    fn exchange_evenly(&self) -> Stream<S, D>;
}

impl<S: Scope, D: ExchangeData> ExchangeEvenly<S, D> for Stream<S, D> {
    fn exchange_evenly(&self) -> Stream<S, D> {
        let peers = self.scope().peers() as u64;
        self.unary(Pipeline, "ExchangeEvenly", |_, _| {
            let mut count = 0_u64;
            move |input, output| {
                input.for_each(|time, data| {
                    output.session(&time).give_iterator(data.drain(..).map(|x| {
                        count += 1;
                        (x, count % peers)
                    }));
                });
            }
        }).exchange(|(_x, n)| *n)
            .map(|(x, _)| x)
    }
}
