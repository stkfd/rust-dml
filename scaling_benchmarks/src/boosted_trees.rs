extern crate flexi_logger;
extern crate log;
extern crate ml_dataflow;
extern crate ndarray;
extern crate timely;
extern crate timely_communication;
extern crate toml;
#[macro_use]
extern crate serde_derive;
extern crate serde;

use flexi_logger::Logger;
use timely::dataflow::Stream;

use self::config::BoostedTrees;
use ml_dataflow::data::dataflow::random::{params::*, RandRegressionTrainingSource};
use ml_dataflow::data::{
    dataflow::error_measures::{MeasurePredictionError, Rmse},
    quantize::*,
    TrainingData,
};
use ml_dataflow::models::decision_tree::regression::*;
use ml_dataflow::models::gradient_boost::GradientBoostingRegression;
use ml_dataflow::models::*;
use ndarray::prelude::*;
use timely::dataflow::operators::*;
use timely::progress::nested::Summary;
use timely::progress::timestamp::RootTimestamp;
use timely_communication::initialize::Configuration;

mod config;

fn main() {
    let config = self::config::read().boosted_trees;
    Logger::with_env_or_str("warn,ml_dataflow::models::gradient_boost=info").start().unwrap();

    println!("Run for stages {:?}", config.stages);
    for stages in config.stages.clone() {
        run(config.clone(), stages, 8)
    }

    println!("Run for threads {:?}", config.threads);
    for threads in config.threads.clone() {
        println!("Run with {} threads", threads);
        run(config.clone(), 25, threads)
    }
}

fn run(config: BoostedTrees, stages: usize, threads: usize) {
    let samples_per_thread = config.samples / threads;
    ::timely::execute(Configuration::Process(threads), move |root| {
        let config = config.clone();
        let distribution_params = arr1(&[
            UniformParams::new(0., 20.),
            UniformParams::new(0., 20.),
            UniformParams::new(0., 20.),
            UniformParams::new(0., 20.),
            UniformParams::new(0., 20.),
        ]);
        let distribution_quantizers: Vec<_> = distribution_params
            .iter()
            .map(|dist_param| {
                UniformQuantizer::from_distribution_params(
                    dist_param,
                    config.clone().quantize_resolution,
                )
            })
            .collect();

        let rand_source = RandRegressionTrainingSource::new(
            move |x: &ArrayView1<f64>, x_mapped: &mut ArrayViewMut1<i64>| {
                for ((i, x_quant), &x) in x_mapped.indexed_iter_mut().zip(x.iter()) {
                    *x_quant = distribution_quantizers[i].quantize(x);
                }
                x[0] * x[0] * 0.6 + x[1] * 0.3 - x[2] * 0.2 + x[3] * 2. + x[4] * 0.7
            },
        ).x_distributions(distribution_params);

        let model = StreamingRegressionTree::new(
            config.levels as u64,
            samples_per_thread as u64,
            config.bins,
            1.0,
        );
        let boost_model =
            GradientBoostingRegression::new(stages as u64, model, config.learning_rate);

        root.dataflow::<u64, _, _>(move |root_scope| {
            let training_stream: Stream<_, TrainingData<i64, f64>> = rand_source
                .clone()
                .samples(samples_per_thread as usize, 1)
                .to_stream(Summary::Local(1), RootTimestamp::new(1), root_scope);

            let boost_chain = training_stream.train_meta(&boost_model);

            let predict_data = rand_source.clone().samples(5000, 1).to_stream(
                Summary::Local(1),
                RootTimestamp::new(1),
                root_scope,
            );

            predict_data
                .map(|t_d| t_d.x)
                .predict(&boost_model, boost_chain.broadcast())
                .map(|res| res.expect("prediction"))
                .prediction_error(&predict_data.map(|t_d| t_d.y), Rmse)
                .inspect(move |rmse| println!("{} Stages, RMSE: {}", stages, rmse));
        });
        while root.step() {}
    }).expect("Execute dataflow");
}
