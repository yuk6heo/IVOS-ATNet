import os

class Config(object):
    def __init__(self):

        ################################ C ##################################
        # DAVIS path
        self.davis_dataset_dir = '/home/yuk/data_ssd/datasets/DAVIS'
        self.test_gpu_id = 2
        self.test_metric_list = ['J', 'J_AND_F']

        ################################ For test parameters ##################################
        self.test_host = 'localhost'  # 'localhost' for subsets train and val.
        self.test_subset = 'val'
        self.test_userkey = None
        self.test_propagation_proportion = 0.99
        self.test_propth = 0.8
        self.test_min_nb_nodes = 2
        self.test_save_all_segs_option = True

        ############################### Other parameters ##################################
        self.mean, self.var = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        self.scribble_dilation_param = 5

        # Rel path
        # project_path = os.path.dirname(__file__)
        # self.font_dir = project_path + '/fonts/'
        self.palette_dir = self.davis_dataset_dir + '/Annotations/480p/bear/00000.png'
        self.test_result_df_dir = 'results/test_result_davisframework'
        self.test_result_rw_dir = 'results/test_result_realworld'
        self.test_load_state_dir = 'ATNet-checkpoint.pth'  # CKpath
