use std::collections::{hash_map::Entry, HashMap};
use std::result::Result as StdResult;
use timely::dataflow::{Scope, Stream};
use timely::dataflow::channels::pact::Pipeline;
use timely::dataflow::operators::Operator;
use timely::Data;

pub mod generic_ops;
pub mod csv;

type Result<T> = StdResult<T, ::failure::Error>;

/// Specifies the data needed to create a DataProvider in a serializable struct that can be sent to
/// other workers. Types that implement this trait can be converted into their respective
/// DataProviders using try_from/try_into.
pub trait DataProviderSpec: Data {
    type Provider: TryFrom<Self, Error=::failure::Error>;
}

/// Fetches data from a DataProvider, borrowing it immutably.
pub trait DataOperator<Out: Data, Provider>: Data + MutDataOperator<Out, Provider> {
    fn fetch_vec(&self, provider: &Provider) -> Result<Vec<Out>>;
    fn fetch_into_iter(&self, provider: &Provider) -> Result<Box<Iterator<Item=Out>>>;
}

/// Fetches data from a DataProvider, modifying its internal state in the process.
pub trait MutDataOperator<Out: Data, Provider>: Data + IntoDataOperator<Out, Provider> {
    fn fetch_vec(&self, provider: &mut Provider) -> Result<Vec<Out>>;
    fn fetch_into_iter(&self, provider: &mut Provider) -> Result<Box<Iterator<Item=Out>>>;
}

//// I think this should work when specialization is more fully implemented. Currently causes a compile error due to conflicting impls
//impl<R: Data, P, T: MutQuery<R, P>> IntoQuery<R, P> for T
//{
//    fn fetch_vec(&self, mut provider: P) -> Result<Vec<R>> {
//        <Self as MutQuery<R, P>>::fetch_vec(self, &mut provider)
//    }
//
//    fn fetch_into_iter(&self, mut provider: P) -> Result<Box<Iterator<Item=R>>> {
//        <Self as MutQuery<R, P>>::fetch_into_iter(self, &mut provider)
//    }
//}

/// Fetches data from a DataProvider and consumes it in the process.
pub trait IntoDataOperator<Out: Data, Provider>: Data {
    fn fetch_vec(&self, provider: Provider) -> Result<Vec<Out>>;
    fn fetch_into_iter(&self, provider: Provider) -> Result<Box<Iterator<Item=Out>>>;
}

pub trait ExecuteDataOperators<Ps: DataProviderSpec, R: Data, Q: IntoDataOperator<R, Ps::Provider>, S: Scope> {
    fn exec_data_ops(&self, provider_spec_stream: Stream<S, Ps>) -> Stream<S, R>;
}

impl<PSpec, Out, Op, S> ExecuteDataOperators<PSpec, Out, Op, S> for Stream<S, Op>
    where PSpec: DataProviderSpec,
          Out: Data,
          Op: IntoDataOperator<Out, PSpec::Provider>,
          S: Scope,
{
    fn exec_data_ops(&self, provider_spec_stream: Stream<S, PSpec>) -> Stream<S, Out> {
        self.binary_frontier(&provider_spec_stream, Pipeline, Pipeline, "fetch_queries", |_| {
            let mut spec_stash: HashMap<_, PSpec> = HashMap::new();
            let mut ops_stash: HashMap<_, Vec<Op>> = HashMap::new();

            move |in_ops, in_specs, out| {
                // stash data operations while waiting for the corresponding DataProviderSpec to arrive
                in_ops.for_each(|time, data| {
                    ops_stash.entry(time.clone())
                        .or_insert_with(|| Vec::new())
                        .extend(data.drain(..));
                });

                // stash arriving DataProviderSpec
                in_specs.for_each(|time, data| {
                    assert_eq!(data.len(), 1);
                    let spec = data.drain(..).next().unwrap();
                    match spec_stash.entry(time.clone()) {
                        Entry::Vacant(e) => e.insert(spec),
                        Entry::Occupied(_) => panic!("Received more than one DataProviderSpec instance for a given time when only one was expected"),
                    };
                });

                // execute operators when their respective DataProviderSpec has been received
                for (time, ops) in ops_stash.iter_mut() {
                    if !in_specs.frontier().less_equal(time.time()) {
                        let mut session = out.session(&time);

                        let spec = spec_stash.get(&time).expect("Didn't receive DataProviderSpec");

                        for op in ops.drain(..) {
                            session.give_iterator(
                                op.fetch_into_iter(
                                    spec.clone().try_into().expect("Failed to create DataProvider")
                                ).expect("Failed to execute data operation").into_iter()
                            );
                        }
                    }
                }

                // drop executed operation stash entries
                ops_stash.retain(|_, v| v.len() > 0);
                // drop no longer needed DataProviderSpecs
                spec_stash.retain(|time, _| in_ops.frontier().less_equal(time.time()))
            }
        })
    }
}
