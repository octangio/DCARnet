from tensorflow.keras import models
import tensorflow as tf
from tensorflow.python.platform import gfile
from os import path, chdir
import pickle
import glob, os

"""
Example:

from my_losses import my_jaccard_loss

keras_model_path = '../logs/model_01_0.23.hdf5'
custom_objects = {'my_jaccard_loss': my_jaccard_loss}
my_model_convert_keras2tensorflow(keras_model_path, custom_objects=custom_objects, out_dir=None, out_pb_filename=None)

"""


def my_get_model_file(logdir, checkpoint_file):
    model_files = os.path.join(logdir, checkpoint_file)
    return model_files


def my_get_callback_files(logdir, file_extents='.pickle'):
    return [os.path.join(logdir, f) for f in os.listdir(logdir) if f.endswith(file_extents)]


def my_get_init_epoch(model_path):
    model_name = os.path.basename(model_path)
    return int(model_name.split('_')[1])


def my_get_init_epoch_lr(logfile):
    with open(logfile, "r") as f:
        strline = f.readline()
    if strline is not None and strline is not '':
        values = strline.split()
        rt = [int(values[0]), float(values[1])]
    else:
        rt = [0, 0.001]
    return rt


def my_load_model(model_func, custom_objects=None, logdir='./logs', checkpoint_file='checkpoint.ckp'):
    ckp_f = os.path.join(logdir, checkpoint_file)
    rt = [0, 0.0003, '']
    if os.path.exists(ckp_f):
        with open(ckp_f, 'r') as f:
            strline = f.readline()
            if len(strline) is not 0 and strline is not '':
                values = strline.split()
                rt = [int(values[0]), float(values[1]), values[2]]
    model_path = os.path.join(logdir, rt[2])
    # Load checkpoint:
    initial_epoch = 0
    if os.path.isfile(model_path):
        # Load model:
        if model_func:
            model_func.load_weights(model_path, custom_objects)
        else:
            model_func = models.load_model(model_path, custom_objects)
        # Finding the epoch index from which we are resuming
        initial_epoch = rt[0]
        initial_lr = rt[1]
        tf.keras.backend.set_value(model_func.optimizer.lr, initial_lr)
    model = model_func

    return model, initial_epoch


class MyModel(object):

    @staticmethod
    def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
        """
        Freezes the state of a session into a pruned computation graph.

        Creates a new computation graph where variable nodes are replaced by
        constants taking their current value in the session. The new graph will be
        pruned so subgraphs that are not necessary to compute the requested
        outputs are removed.
        @param session The TensorFlow session to be frozen.
        @param keep_var_names A list of variable names that should not be frozen,
                              or None to freeze all the variables in the graph.
        @param output_names Names of the relevant graph outputs.
        @param clear_devices Remove the device directives from the graph for better portability.
        @return The frozen graph definition.
        """
        graph = session.graph
        with graph.as_default():
            freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
            output_names = output_names or []
            output_names += [v.op.name for v in tf.global_variables()]
            input_graph_def = graph.as_graph_def()
            if clear_devices:
                for node in input_graph_def.node:
                    node.device = ""
            frozen_graph = tf.graph_util.convert_variables_to_constants(
                session, input_graph_def, output_names, freeze_var_names)
            return frozen_graph

    @staticmethod
    def my_model_convert_keras2tensorflow(keras_model_path, custom_objects=None, out_dir=None, out_pb_filename=None):
        if out_dir is None:
            out_dir = path.dirname(keras_model_path)
        if out_pb_filename is None:
            base_name = path.basename(keras_model_path)
            out_pb_filename = base_name + '_tf.pb'
        model = models.load_model(keras_model_path, custom_objects=custom_objects)
        model.summary()
        frozen_graph = MyModel.freeze_session(backend.get_session(),
                                              output_names=[out.op.name for out in model.outputs])
        tf.train.write_graph(frozen_graph, out_dir, out_pb_filename, as_text=False)

    @staticmethod
    def my_save_tensor_name_to_file(session, out_dir):
        infofp = open(path.join(out_dir, 'info.txt'), 'w')
        for n in session.graph.as_graph_def().node:
            infofp.write(n.name + '\n')
        infofp.close()

    @staticmethod
    def my_save_graph_to_tensorboard(session, out_dir):
        writer = tf.summary.FileWriter(out_dir, graph=session.graph)
        writer.flush()
        writer.close()

    @staticmethod
    def my_load_model_from_pb_file(sesssion, pb_file):
        with gfile.FastGFile(pb_file, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            sesssion.graph.as_default()
            tf.import_graph_def(graph_def)
            return sesssion

