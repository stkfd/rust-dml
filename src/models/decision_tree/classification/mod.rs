#![allow(unknown_lints)]

pub mod histogram;
pub mod impurity;
mod split_leaves;
mod streaming_classification_tree;

pub use self::streaming_classification_tree::StreamingClassificationTree;

/*trait Predict<S: Scope, In: Data, P: Data> {
    fn predict(inputs: Stream<S, In>) -> Stream<S, P>;
}*/
