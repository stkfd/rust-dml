extern crate flexi_logger;
extern crate log;
extern crate ml_dataflow;
extern crate ndarray;
extern crate timely;
extern crate timely_communication;

use flexi_logger::Logger;
use timely::dataflow::Stream;

use ml_dataflow::data::dataflow::random::{params::*, RandRegressionTrainingSource};
use ml_dataflow::data::{
    dataflow::error_measures::{MeasurePredictionError, Rmse},
    quantize::*,
    TrainingData,
};
use ml_dataflow::models::decision_tree::regression::*;
use ml_dataflow::models::*;
use ndarray::prelude::*;
use timely::dataflow::operators::*;
use timely::progress::nested::Summary;
use timely::progress::timestamp::RootTimestamp;
use timely_communication::initialize::Configuration;

fn main() {
    let tree_levels = 8;
    let n_threads = 8;
    let n_samples = 2_000_000;

    let quantize_resolution = 20;
    let histogram_bins = 50;

    let samples_per_thread = n_samples / n_threads;

    Logger::with_env_or_str("ml_dataflow=info").start().unwrap();

    ::timely::execute(Configuration::Process(n_threads as usize), move |root| {
        let distribution_params = arr1(&[
            NormalParams::new(0., 5.),
            NormalParams::new(5., 5.),
            NormalParams::new(10., 5.),
        ]);
        let distribution_quantizers = [
            NormalQuantizer::new(0., 5., quantize_resolution),
            NormalQuantizer::new(5., 5., quantize_resolution),
            NormalQuantizer::new(10., 5., quantize_resolution),
        ];

        let rand_source = RandRegressionTrainingSource::new(
            move |x: &ArrayView1<f64>, x_mapped: &mut ArrayViewMut1<i64>| {
                for ((i, x_quant), &x) in x_mapped.indexed_iter_mut().zip(x.iter()) {
                    *x_quant = distribution_quantizers[i].quantize(x);
                }
                x[0] * 0.6 + x[1] * 0.3 - x[2] * 0.2
            },
        ).x_distributions(distribution_params);

        let model = StreamingRegressionTree::new(tree_levels, n_samples, histogram_bins, 1.0);

        root.dataflow::<u64, _, _>(|root_scope| {
            let worker = root_scope.index();

            let training_stream: Stream<_, TrainingData<i64, f64>> = rand_source
                .clone()
                .samples(samples_per_thread as usize, 1)
                .to_stream(Summary::Local(1), RootTimestamp::new(1), root_scope);

            let trees = training_stream.train(&model);

            let predict_data = if worker == 0 {
                rand_source.clone().samples(200, 1).to_stream(
                    Summary::Local(1),
                    RootTimestamp::new(1),
                    root_scope,
                )
            } else {
                vec![].to_stream(root_scope)
            }.inspect(|d| println!("{}", d.y()));

            predict_data
                .map(|t_d| t_d.x)
                .predict(&model, trees.broadcast())
                .map(|res| res.expect("prediction"))
                .prediction_error(&predict_data.map(|t_d| t_d.y), Rmse)
                .inspect_time(move |time, d| {
                    println!("RMSE {:?} {}", time, d);
                });
        });
        while root.step() {}
    }).expect("Execute dataflow");
}
