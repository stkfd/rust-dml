extern crate flexi_logger;
extern crate log;
extern crate ml_dataflow;
extern crate ndarray;
extern crate quicli;
extern crate timely;
extern crate timely_communication;
#[macro_use]
extern crate serde_derive;
extern crate serde;
#[macro_use]
extern crate lazy_static;
extern crate csv;
extern crate toml;

use flexi_logger::Logger;
use timely::dataflow::Scope;
use timely::dataflow::Stream;

use ml_dataflow::data::dataflow::random::{params::*, RandRegressionTrainingSource};
use ml_dataflow::data::{
    dataflow::error_measures::{MeasurePredictionError, Rmse},
    providers::csv_stream::CsvTrainingDataSource,
    quantize::*,
    TrainingData, TrainingDataSample,
};
use ml_dataflow::models::decision_tree::regression::*;
use ml_dataflow::models::*;
use ndarray::prelude::*;
use quicli::prelude::*;
use std::fs::File;
use std::io::prelude::*;
use timely::dataflow::operators::*;
use timely::progress::nested::Summary;
use timely::progress::timestamp::RootTimestamp;
use timely_communication::initialize::Configuration;

fn main() {
    let args = Cli::from_args();

    // read and parse config file
    let config: Config = {
        let mut config_file = File::open(args.config).expect("Config file not found");
        let mut contents = String::new();
        config_file
            .read_to_string(&mut contents)
            .expect("something went wrong reading the config file");
        ::toml::from_str(&contents).expect("Config file format contained errors")
    };

    if args.generate_data {
        generate_data(args.process_id, config);
    } else {
        Logger::with_env_or_str("ml_dataflow=warn").start().unwrap();
        train(args.process_id, config);
    }
}

lazy_static! {
    static ref DIST_PARAMS: Array1<NormalParams> = arr1(&[
        NormalParams::new(0., 5.),
        NormalParams::new(5., 5.),
        NormalParams::new(10., 5.),
    ]);
}

fn train(process_id: usize, config: Config) {
    let host_addresses: Vec<_> = config
        .cluster
        .hosts
        .iter()
        .cloned()
        .map(|host_config| host_config.address)
        .collect();

    ::timely::execute(
        Configuration::Cluster(config.cluster.workers, process_id, host_addresses, true),
        move |root| {
            let config = config.clone();
            let worker_local_thread_index = root.index() - (config.cluster.workers * process_id);
            let path = config.cluster.hosts[process_id].threads[worker_local_thread_index]
                .data_path
                .clone();
            
            let model = StreamingRegressionTree::new(
                config.model.levels,
                config.model.points_per_worker,
                config.model.bins,
                1.0,
            );

            root.dataflow::<u64, _, _>(move |root_scope| {
                let worker = root_scope.index();
                println!("worker {}", worker);

                let quantizers: Vec<_> = DIST_PARAMS
                    .iter()
                    .map(|dist_param| {
                        NormalQuantizer::from_distribution_params(
                            dist_param,
                            config.model.quantize_resolution,
                        )
                    })
                    .collect();

                let training_stream: Stream<_, TrainingData<i64, f64>> = root_scope
                    .training_data_from_csv(path, config.model.points_per_worker as usize)
                    .map(move |training_data| {
                        let mut x_mapped =
                            unsafe { Array2::<i64>::uninitialized(training_data.x().dim()) };
                        for (x_row, mut x_mapped_row) in training_data
                            .x()
                            .outer_iter()
                            .zip(x_mapped.outer_iter_mut())
                        {
                            for (i, x_i) in x_row.indexed_iter() {
                                x_mapped_row[i] = quantizers[i].quantize(*x_i);
                            }
                        }
                        TrainingData {
                            x: x_mapped.into(),
                            y: training_data.y,
                        }
                    });

                let trees = training_stream.train(&model);

                training_stream
                    .map(|t_d| t_d.x)
                    .predict(&model, trees.broadcast())
                    .map(|res| res.expect("prediction"))
                    .prediction_error(&training_stream.map(|t_d| t_d.y), Rmse);
            });
            while root.step() {}
        },
    ).expect("Execute dataflow");
}

fn generate_data(process_id: usize, config: Config) {
    ::timely::execute(
        Configuration::Process(config.cluster.workers),
        move |root| {
            let config = config.clone();
            let thread_index = root.index();
            let path = config.cluster.hosts[process_id].threads[thread_index]
                .data_path
                .clone();

            let rand_source = RandRegressionTrainingSource::new(
                move |x: &ArrayView1<f64>, x_mapped: &mut ArrayViewMut1<f64>| {
                    x_mapped.assign(x);
                    x[0] * x[0] * 0.1 + x[1] * 0.5 - x[2] * 2.
                },
            ).x_distributions(DIST_PARAMS.clone());

            root.dataflow::<u64, _, _>(move |root_scope| {
                rand_source
                    .samples(config.model.points_per_worker as usize, 1)
                    .to_stream(Summary::Local(1), RootTimestamp::new(1), root_scope)
                    .inspect(move |data| {
                        let file = File::create(path.clone()).expect("Open CSV file");
                        let mut wtr = csv::Writer::from_writer(file);
                        for (x, y) in data.x().outer_iter().zip(data.y().iter()) {
                            wtr.serialize(TrainingDataSample::from((x, y)))
                                .expect("serialize training data");
                        }
                        wtr.flush().expect("Write csv");
                    });
            });
            while root.step() {}
        },
    ).expect("Execute dataflow");
}

#[derive(Debug, StructOpt)]
struct Cli {
    /// Number of this process in the cluster
    #[structopt(short = "p", long = "process", default_value = "0")]
    process_id: usize,
    /// Path to the configuration file
    #[structopt(
        short = "c",
        long = "config",
        default_value = "./run_config.toml",
    )]
    config: String,
    /// Generate data only and save it to this path
    #[structopt(long = "generate-data")]
    generate_data: bool,
}

#[derive(Deserialize, Clone)]
struct Config {
    cluster: ClusterConfig,
    model: ModelConfig,
}

#[derive(Deserialize, Clone)]
struct ClusterConfig {
    hosts: Vec<HostConfig>,
    workers: usize,
}

#[derive(Deserialize, Clone)]
struct HostConfig {
    address: String,
    threads: Vec<ThreadConfig>,
}

#[derive(Deserialize, Clone)]
struct ThreadConfig {
    data_path: String,
}

#[derive(Deserialize, Clone)]
struct ModelConfig {
    levels: u64,
    bins: usize,
    points_per_worker: u64,
    trim_ratio: f64,
    quantize_resolution: usize,
}
