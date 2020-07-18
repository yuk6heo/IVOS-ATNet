from davisinteractive.session import DavisInteractiveSession
from davisinteractive import utils as interactive_utils
from davisinteractive.dataset import Davis
from davisinteractive.metrics import batched_jaccard

from libs import custom_transforms as tr, davis2017_torchdataset
import os

import numpy as np
from PIL import Image
import csv
from datetime import datetime

import torch
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import DataLoader

from libs import utils, utils_torch
from libs.analyze_report import analyze_summary
from config import Config
from networks.atnet import ATnet


class Main_tester(object):
    def __init__(self, config):
        self.config = config
        self.Davisclass = Davis(self.config.davis_dataset_dir)
        self.current_time = datetime.now().strftime('%Y%m%d-%H%M%S')
        self._palette = Image.open(self.config.palette_dir).getpalette()
        self.save_res_dir = str()
        self.save_log_dir = str()
        self.save_logger = None
        self.save_csvsummary_dir = str()

        self.net = ATnet()
        self.net.cuda()
        self.net.eval()
        self.net.load_state_dict(torch.load(self.config.test_load_state_dir))

        # To implement ordered test
        self.scr_indices = [1, 2, 3]
        self.max_nb_interactions = 8
        self.max_time = self.max_nb_interactions * 30
        self.scr_samples = []
        for v in sorted(self.Davisclass.sets[self.config.test_subset]):
            for idx in self.scr_indices:
                self.scr_samples.append((v, idx))

        self.img_size, self.num_frames, self.n_objects, self.final_masks, self.tmpdict_siact = None, None, None, None, None
        self.pad_info, self.hpad1, self.wpad1, self.hpad2, self.wpad2 = None, None, None, None, None

    def run_for_diverse_metrics(self, ):

        with torch.no_grad():
            for metric in self.config.test_metric_list:
                if metric == 'J':
                    dir_name = os.path.split(os.path.split(__file__)[0])[1] + '[J]_' + self.current_time
                elif metric == 'J_AND_F':
                    dir_name = os.path.split(os.path.split(__file__)[0])[1] + '[JF]_' + self.current_time
                else:
                    dir_name = None
                    print("Impossible metric is contained in config.test_metric_list!")
                    raise NotImplementedError()
                self.save_res_dir = os.path.join(self.config.test_result_dir, dir_name)
                utils.mkdir(self.save_res_dir)
                self.save_csvsummary_dir = os.path.join(self.save_res_dir, 'summary_in_csv.csv')
                self.save_log_dir = os.path.join(self.save_res_dir, 'test_logs.txt')
                self.save_logger = utils.logger(self.save_log_dir)
                self.save_logger.printNlog(dir_name)
                curr_path = os.path.dirname(os.path.abspath(__file__))
                os.system('cp {}/config.py {}/config.py'.format(curr_path, self.save_res_dir))



                self.run_IVOS(metric)

    def run_IVOS(self, metric):
        seen_seq = {}
        numseq, tmpseq = 0, ''
        output_dict = dict()
        output_dict['average_objs_iou'] = dict()
        output_dict['average_iact_iou'] = np.zeros(self.max_nb_interactions)
        output_dict['annotated_frames'] = dict()

        with open(self.save_csvsummary_dir, mode='a') as csv_file:
            writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(['sequence', 'obj_idx', 'scr_idx'] + ['round-' + str(i + 1) for i in range(self.max_nb_interactions)])

        with DavisInteractiveSession(host=self.config.test_host,
                                     user_key=self.config.test_userkey,
                                     davis_root=self.config.davis_dataset_dir,
                                     subset=self.config.test_subset,
                                     report_save_dir=self.save_res_dir,
                                     max_nb_interactions=self.max_nb_interactions,
                                     max_time=self.max_time,
                                     metric_to_optimize=metric) as sess:

            sess.connector.service.robot.min_nb_nodes = self.config.test_min_nb_nodes
            sess.samples = self.scr_samples
            # sess.samples = [('dog', 3)]

            while sess.next():
                # Get the current iteration scribbles
                self.sequence, scribbles, first_scribble = sess.get_scribbles(only_last=False)

                if first_scribble:
                    anno_dict = {'frames': [], 'annotated_masks': [], 'masks_tobe_modified': []}
                    n_interaction = 1
                    info = Davis.dataset[self.sequence]
                    self.img_size = info['image_size'][::-1]
                    self.num_frames = info['num_frames']
                    self.n_objects = info['num_objects']
                    info = None
                    seen_seq[self.sequence] = 1 if self.sequence not in seen_seq.keys() else seen_seq[self.sequence] + 1
                    scr_id = seen_seq[self.sequence]
                    self.final_masks = np.zeros([self.num_frames, self.img_size[0], self.img_size[1]])
                    self.pad_info = utils.apply_pad(self.final_masks[0])[1]
                    self.hpad1, self.wpad1 = self.pad_info[0][0], self.pad_info[1][0]
                    self.hpad2, self.wpad2 = self.pad_info[0][1], self.pad_info[1][1]
                    self.h_ds, self.w_ds = int((self.img_size[0] + sum(self.pad_info[0])) / 4), int((self.img_size[1] + sum(self.pad_info[1])) / 4)
                    self.anno_6chEnc_r5_list = []
                    self.anno_3chEnc_r5_list = []
                    self.prob_map_of_frames = torch.zeros((self.num_frames, self.n_objects, 4 * self.h_ds, 4 * self.w_ds)).cuda()
                    self.gt_masks = self.Davisclass.load_annotations(self.sequence)

                    IoU_over_eobj = []

                else:
                    n_interaction += 1

                self.save_logger.printNlog('\nRunning sequence {} in (scribble index: {}) (round: {})'
                                           .format(self.sequence, sess.samples[sess.sample_idx][1], n_interaction))

                annotated_now = interactive_utils.scribbles.annotated_frames(sess.sample_last_scribble)[0]
                anno_dict['frames'].append(annotated_now)  # Where we save annotated frames
                anno_dict['masks_tobe_modified'].append(self.final_masks[annotated_now])  # mask before modefied at the annotated frame

                # Get Predicted mask & Mask decision from pred_mask
                self.final_masks = self.run_VOS_singleiact(n_interaction, scribbles, anno_dict['frames'])  # self.final_mask changes

                if self.config.test_save_all_segs_option:
                    utils.mkdir(
                        os.path.join(self.save_res_dir, 'result_video', '{}-scr{:02d}/round{:02d}'.format(self.sequence, scr_id, n_interaction)))
                    for fr in range(self.num_frames):
                        savefname = os.path.join(self.save_res_dir, 'result_video',
                                                 '{}-scr{:02d}/round{:02d}'.format(self.sequence, scr_id, n_interaction),
                                                 '{:05d}.png'.format(fr))
                        tmpPIL = Image.fromarray(self.final_masks[fr].astype(np.uint8), 'P')
                        tmpPIL.putpalette(self._palette)
                        tmpPIL.save(savefname)

                # Submit your prediction
                sess.submit_masks(self.final_masks)  # F, H, W

                # print sequence name
                if tmpseq != self.sequence:
                    tmpseq, numseq = self.sequence, numseq + 1
                print(str(numseq) + ':' + str(self.sequence) + '-' + str(seen_seq[self.sequence]) + '\n')

                ## Visualizers and Saver
                # IoU estimation
                jaccard = batched_jaccard(self.gt_masks,
                                          self.final_masks,
                                          average_over_objects=False,
                                          nb_objects=self.n_objects
                                          )  # frames, objid

                IoU_over_eobj.append(jaccard)

                anno_dict['annotated_masks'].append(self.final_masks[annotated_now])  # mask after modefied at the annotated frame

                if self.max_nb_interactions == len(anno_dict['frames']):  # After Lastround -> total 90 iter
                    seq_scrid_name = self.sequence + str(scr_id)

                    # IoU manager
                    IoU_over_eobj = np.stack(IoU_over_eobj, axis=0)  # niact,frames,n_obj
                    IoUeveryround_perobj = np.mean(IoU_over_eobj, axis=1)  # niact,n_obj
                    output_dict['average_iact_iou'] += np.sum(IoU_over_eobj[list(range(n_interaction)), anno_dict['frames']], axis=-1)
                    output_dict['annotated_frames'][seq_scrid_name] = anno_dict['frames']

                    # write csv
                    for obj_idx in range(self.n_objects):
                        with open(self.save_csvsummary_dir, mode='a') as csv_file:
                            writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                            writer.writerow([self.sequence, str(obj_idx + 1), str(scr_id)] + list(IoUeveryround_perobj[:, obj_idx]))

        summary = sess.get_global_summary(save_file=self.save_res_dir + '/summary_' + sess.report_name[7:] + '.json')
        analyze_summary(self.save_res_dir + '/summary_' + sess.report_name[7:] + '.json', metric=metric)

        # final_IOU = summary['curve'][metric][-1]
        average_IoU_per_round = summary['curve'][metric][1:-1]

        torch.cuda.empty_cache()
        model = None
        return average_IoU_per_round

    def run_VOS_singleiact(self, n_interaction, scribbles_data, annotated_frames):

        annotated_frames_np = np.array(annotated_frames)
        num_workers = 4
        annotated_now = annotated_frames[-1]
        scribbles_list = scribbles_data['scribbles']
        seq_name = scribbles_data['sequence']

        output_masks = self.final_masks.copy().astype(np.float64)

        prop_list = utils.get_prop_list(annotated_frames, annotated_now, self.num_frames, proportion=self.config.test_propagation_proportion)
        prop_fore = sorted(prop_list)[0]
        prop_rear = sorted(prop_list)[-1]

        # Interaction settings
        pm_ps_ns_3ch_t = []  # n_obj,3,h,w
        if n_interaction == 1:
            for obj_id in range(1, self.n_objects + 1):
                pos_scrimg = utils.scribble_to_image(scribbles_list, annotated_now, obj_id,
                                                            dilation=self.config.scribble_dilation_param,
                                                            prev_mask=self.final_masks[annotated_now])
                pm_ps_ns_3ch_t.append(np.stack([np.ones_like(pos_scrimg) / 2, pos_scrimg, np.zeros_like(pos_scrimg)], axis=0))
            pm_ps_ns_3ch_t = np.stack(pm_ps_ns_3ch_t, axis=0)  # n_obj,3,h,w
            # Image.fromarray((scr_img[:, :, 1] * 255).astype(np.uint8)).save('/home/six/Desktop/CVPRW_figure/judo_obj1_scr.png')

        else:
            for obj_id in range(1, self.n_objects + 1):
                prev_round_input = (self.final_masks[annotated_now] == obj_id).astype(np.float32)  # H,W
                pos_scrimg, neg_scrimg = utils.scribble_to_image(scribbles_list, annotated_now, obj_id,
                                                                        dilation=self.config.scribble_dilation_param,
                                                                        prev_mask=self.final_masks[annotated_now], blur=True,
                                                                        singleimg=False, seperate_pos_neg=True)
                pm_ps_ns_3ch_t.append(np.stack([prev_round_input, pos_scrimg, neg_scrimg], axis=0))
            pm_ps_ns_3ch_t = np.stack(pm_ps_ns_3ch_t, axis=0)  # n_obj,3,h,w
        pm_ps_ns_3ch_t = torch.from_numpy(pm_ps_ns_3ch_t).cuda()

        if (prop_list[0] != annotated_now) and (prop_list.count(annotated_now) != 2):
            print(str(prop_list))
            raise NotImplementedError
        print(str(prop_list))  # we made our proplist first backward, and then forward

        composed_transforms = transforms.Compose([tr.Normalize_ApplymeanvarImage(self.config.mean, self.config.var),
                                                  tr.ToTensor()])
        db_test = davis2017_torchdataset.DAVIS2017(split='val', transform=composed_transforms, root=self.config.davis_dataset_dir,
                                                   custom_frames=prop_list, seq_name=seq_name, rgb=True,
                                                   obj_id=None, no_gt=True, retname=True, prev_round_masks=self.final_masks, )
        testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=True)

        flag = 0  # 1: propagating backward, 2: propagating forward
        print('[{:01d} round] processing...'.format(n_interaction))

        for ii, batched in enumerate(testloader):
            # batched : image, scr_img, 0~fr, meta
            inpdict = dict()
            operating_frame = int(batched['meta']['frame_id'][0])

            for inp in batched:
                if inp == 'meta': continue
                inpdict[inp] = Variable(batched[inp]).cuda()

            inpdict['image'] = inpdict['image'].expand(self.n_objects, -1, -1, -1)

            #################### Iaction ########################
            if operating_frame == annotated_now:  # Check the round is on interaction
                if flag == 0:
                    flag += 1
                    adjacent_to_anno = True
                elif flag == 1:
                    flag += 1
                    adjacent_to_anno = True
                    continue
                else:
                    raise NotImplementedError

                pm_ps_ns_3ch_t = torch.nn.ReflectionPad2d(self.pad_info[1] + self.pad_info[0])(pm_ps_ns_3ch_t)
                inputs = torch.cat([inpdict['image'], pm_ps_ns_3ch_t], dim=1)
                output_logit, anno_6chEnc_r5 = self.net.forward_ANet(inputs)  # [nobj, 1, P_H, P_W], # [n_obj,2048,h/16,w/16]
                output_prob_anno = torch.sigmoid(output_logit)
                prob_onehot_t = output_prob_anno[:, 0].detach()

                anno_3chEnc_r5, _, _, r2_prev_fromanno = self.net.encoder_3ch.forward(inpdict['image'])
                self.anno_6chEnc_r5_list.append(anno_6chEnc_r5)
                self.anno_3chEnc_r5_list.append(anno_3chEnc_r5)

                if len(self.anno_6chEnc_r5_list) != len(annotated_frames):
                    raise NotImplementedError



            #################### Propagation ########################
            else:
                # Flag [1: propagating backward, 2: propagating forward]
                if adjacent_to_anno:
                    r2_prev = r2_prev_fromanno
                    predmask_prev = output_prob_anno
                else:
                    predmask_prev = output_prob_prop
                adjacent_to_anno = False

                output_logit, r2_prev = self.net.forward_TNet(
                    self.anno_3chEnc_r5_list, inpdict['image'], self.anno_6chEnc_r5_list, r2_prev, predmask_prev)  # [nobj, 1, P_H, P_W]
                output_prob_prop = torch.sigmoid(output_logit)
                prob_onehot_t = output_prob_prop[:, 0].detach()

                smallest_alpha = 0.5
                if flag == 1:
                    sorted_frames = annotated_frames_np[annotated_frames_np < annotated_now]
                    if len(sorted_frames) == 0:
                        alpha = 1
                    else:
                        closest_addianno_frame = np.max(sorted_frames)
                        alpha = smallest_alpha + (1 - smallest_alpha) * (
                                (operating_frame - closest_addianno_frame) / (annotated_now - closest_addianno_frame))
                else:
                    sorted_frames = annotated_frames_np[annotated_frames_np > annotated_now]
                    if len(sorted_frames) == 0:
                        alpha = 1
                    else:
                        closest_addianno_frame = np.min(sorted_frames)
                        alpha = smallest_alpha + (1 - smallest_alpha) * (
                                (closest_addianno_frame - operating_frame) / (closest_addianno_frame - annotated_now))

                prob_onehot_t = (alpha * prob_onehot_t) + ((1 - alpha) * self.prob_map_of_frames[operating_frame])

            # Final mask indexing
            self.prob_map_of_frames[operating_frame] = prob_onehot_t

        output_masks[prop_fore:prop_rear + 1] = \
            utils_torch.combine_masks_with_batch(self.prob_map_of_frames[prop_fore:prop_rear + 1],
                n_obj=self.n_objects, th=self.config.test_propth
            )[:, 0, self.hpad1:-self.hpad2, self.wpad1:-self.wpad2].cpu().numpy().astype(np.float)  # f,h,w

        torch.cuda.empty_cache()

        return output_masks


if __name__ == '__main__':
    config = Config()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.test_gpu_id)

    tester = Main_tester(config)
    tester.run_for_diverse_metrics()

    # try:main_val(model,
    #              Config,
    #              min_nb_nodes= min_nb_nodes,
    #              simplyfied_testset= simplyfied_test,tr(config.test_gpu_id)
    #              metric = metric)
    # except: continue
