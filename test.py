from action_predict import action_prediction
from pie_data import PIE
from jaad_data import JAAD
import os, sys, yaml, tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
assert len(gpus) > 0, "Not enough GPU hardware devices available"
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
    tf.config.experimental.set_virtual_device_configuration(
        gpu, [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)]
    )

def test_model(saved_files_path):
    # Load configs
    with open(os.path.join(saved_files_path, 'configs.yaml'), 'r') as f:
        opts = yaml.safe_load(f)
    model_opts = opts['model_opts']
    data_opts  = opts['data_opts']
    net_opts   = opts['net_opts']

    # Build test data
    tte = model_opts['time_to_event'] if isinstance(model_opts['time_to_event'], int) else model_opts['time_to_event'][1]
    data_opts['min_track_size'] = model_opts['obs_length'] + tte
    if model_opts['dataset'] == 'pie':
        imdb = PIE(data_path='/home/zzhonghang/Pedestrian_Crossing_Intention_Prediction/data/pie')
    elif model_opts['dataset'] == 'jaad':
        imdb = JAAD(data_path='/home/zzhonghang/Pedestrian_Crossing_Intention_Prediction/JAAD')
    else:
        raise ValueError(f"{model_opts['dataset']} dataset is incorrect")


    method_class = action_prediction(model_opts['model'])(**net_opts)
    beh_seq_test = imdb.generate_data_trajectory_sequence('test', **data_opts)


    model_file = 'model.h5'
    print(f"\nTesting {model_file} ...")
    acc, auc, f1, precision, recall = method_class.test(
        beh_seq_test, model_path=saved_files_path, model_file=model_file
    )
    print(f"acc:{acc:.4f} auc:{auc:.4f} f1:{f1:.4f} precision:{precision:.4f} recall:{recall:.4f}")

if __name__ == '__main__':
    saved_files_path = sys.argv[1]
    test_model(saved_files_path)
