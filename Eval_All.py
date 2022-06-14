import torch
import numpy as np
from STRESS.network import blocked_AUTOCODER, Predictor, blocked_AUTOCODER_pr
import time
import matplotlib.pyplot as plt
import cv2
from COORD.network import AUTOCODER_array, Predictor


''' Used in Stress Prediction Model to provide shape for stress visualization'''
class coord_predictor():
    def __init__(self):
        dir = r'D:\Guowen\DigitTwinRubber_3D\COORD\\'

        self.Code_predictor_bm = Predictor(code_length=24, params_length=3, hiden_length=64).cuda()
        self.Code_predictor_nb = Predictor(code_length=24, params_length=3, hiden_length=64).cuda()
        self.Code_predictor_pr = Predictor(code_length=24, params_length=3, hiden_length=64).cuda()
        self.autocoder_bm = AUTOCODER_array(vect_length=15184, code_length=24, num_layer=3).cuda()
        self.autocoder_pr = AUTOCODER_array(vect_length=26016, code_length=24, num_layer=3).cuda()
        self.autocoder_nb = AUTOCODER_array(vect_length=13120, code_length=24, num_layer=3).cuda()

        checkpoint = torch.load(dir + "BM_coder_best.pth")
        self.autocoder_bm.load_state_dict(checkpoint)
        self.autocoder_bm.eval()
        checkpoint = torch.load(dir + "PR_coder_best.pth")
        self.autocoder_pr.load_state_dict(checkpoint)
        self.autocoder_pr.eval()
        checkpoint = torch.load(dir + "NB_coder_best.pth")
        self.autocoder_nb.load_state_dict(checkpoint)
        self.autocoder_nb.eval()
        checkpoint = torch.load(dir + "BM_Predictor_last.pth")
        self.Code_predictor_bm.load_state_dict(checkpoint)
        self.Code_predictor_bm.eval()
        checkpoint = torch.load(dir + "PR_Predictor_last.pth")
        self.Code_predictor_pr.load_state_dict(checkpoint)
        self.Code_predictor_pr.eval()
        checkpoint = torch.load(dir + "NB_Predictor_last.pth")
        self.Code_predictor_nb.load_state_dict(checkpoint)
        self.Code_predictor_nb.eval()

    def runcase(self, inputs):
        code_bm = self.Code_predictor_bm(inputs)
        code_nb = self.Code_predictor_nb(inputs)
        code_pr = self.Code_predictor_pr(inputs)
        decoded_output_bm = self.autocoder_bm(code_bm, coder=1)
        decoded_output_pr = self.autocoder_pr(code_pr, coder=1)
        decoded_output_nb = self.autocoder_nb(code_nb, coder=1)
        vect_outputs_bm = decoded_output_bm[0, :].detach().cpu().numpy()
        vect_outputs_pr = decoded_output_pr[0, :].detach().cpu().numpy()
        vect_outputs_nb = decoded_output_nb[0, :].detach().cpu().numpy()
        return vect_outputs_bm, vect_outputs_pr, vect_outputs_nb


'''To show stress in corresponding shape(Predicted by COORD-Predictor)'''
class plot_contour():
    def __init__(self):
        pass

    def refresh(self, vect_outputs, coord_vect, inputs_num, order=0):

        s11 = vect_outputs[0, :].tolist()
        s22 = vect_outputs[1, :].tolist()
        s33 = vect_outputs[2, :].tolist()
        s12 = vect_outputs[3, :].tolist()
        s13 = vect_outputs[4, :].tolist()
        s23 = vect_outputs[5, :].tolist()

        x_coord = coord_vect[0, :]
        y_coord = coord_vect[1, :]
        z_coord = coord_vect[2, :]
        #

        fig = plt.figure(1, dpi=96, figsize=(16, 10))
        scale = 1.1

        '''S11'''
        namestring = "Sxx"
        if order == 0:
            self.vmin_1 = min(s11) * scale
            self.vmax_1 = max(s11) * scale

        ax = fig.add_subplot(231, projection='3d')
        ax.view_init(30, 30)
        plt.gca().set_box_aspect((8, 30, 25))
        p1 = ax.scatter(z_coord, x_coord, y_coord, s=2, cmap="jet", c=s11, label="Predicted " + namestring + " (MPa)")
        p1.set_clim(self.vmin_1, self.vmax_1)
        fig.colorbar(p1, fraction=0.045, pad=0.05)
        ax.set_xlabel('X', fontsize=10)
        ax.set_ylabel('Y', fontsize=10)
        ax.set_zlabel('Z', fontsize=10)
        plt.title("Predicted Stress " + namestring)

        '''S22'''
        namestring = "Syy"
        if order == 0:
            self.vmin_2 = min(s22) * scale
            self.vmax_2 = max(s22) * scale
        ax2 = fig.add_subplot(232, projection='3d')
        ax2.view_init(30, 30)
        plt.gca().set_box_aspect((8, 30, 25))

        p2 = ax2.scatter(z_coord, x_coord, y_coord, s=2, cmap="jet", c=s22, label="Predicted " + namestring + " (MPa)")
        p2.set_clim(self.vmin_2, self.vmax_2)
        fig.colorbar(p2, fraction=0.045, pad=0.05)
        ax2.set_xlabel('X', fontsize=10)
        ax2.set_ylabel('Y', fontsize=10)
        ax2.set_zlabel('Z', fontsize=10)
        plt.title("Predicted Stress " + namestring)

        '''S33'''
        namestring = "Szz"
        if order == 0:
            self.vmin_3 = min(s33)
            self.vmax_3 = max(s33)
        ax3 = fig.add_subplot(233, projection='3d')
        ax3.view_init(30, 30)
        plt.gca().set_box_aspect((8, 30, 25))
        p3 = ax3.scatter(z_coord, x_coord, y_coord, s=2, cmap="jet", c=s33, label="Predicted " + namestring + " (MPa)")
        p3.set_clim(self.vmin_3, self.vmax_3)
        fig.colorbar(p3, fraction=0.045, pad=0.05)
        ax3.set_xlabel('X', fontsize=10)
        ax3.set_ylabel('Y', fontsize=10)
        ax3.set_zlabel('Z', fontsize=10)
        plt.title("Predicted Stress " + namestring)


        '''S12'''
        namestring = "Sxy"
        if order == 0:
            self.vmin_4 = min(s12)
            self.vmax_4 = max(s12)
        ax4 = fig.add_subplot(234, projection='3d')
        ax4.view_init(30, 30)
        plt.gca().set_box_aspect((8, 30, 25))
        p4 = ax4.scatter(z_coord, x_coord, y_coord, s=2, cmap="jet", c=s12, label="Predicted " + namestring + " (MPa)")
        p4.set_clim(self.vmin_4, self.vmax_4)
        fig.colorbar(p4, fraction=0.045, pad=0.05)
        ax4.set_xlabel('X', fontsize=10)
        ax4.set_ylabel('Y', fontsize=10)
        ax4.set_zlabel('Z', fontsize=10)
        plt.title("Predicted Stress "+namestring)


        '''S13'''
        namestring = "Sxz"
        if order == 0:
            self.vmin_5 = min(s13)
            self.vmax_5 = max(s13)
        ax5 = fig.add_subplot(235, projection='3d')
        ax5.view_init(30, 30)
        plt.gca().set_box_aspect((8, 30, 25))
        p5 = ax5.scatter(z_coord, x_coord, y_coord, s=2, cmap="jet", c=s13, label="Predicted " + namestring + " (MPa)")
        p5.set_clim(self.vmin_5, self.vmax_5)
        fig.colorbar(p5, fraction=0.045, pad=0.05)
        ax5.set_xlabel('X', fontsize=10)
        ax5.set_ylabel('Y', fontsize=10)
        ax5.set_zlabel('Z', fontsize=10)
        plt.title("Predicted Stress " + namestring)


        '''S23'''
        namestring = "Syz"
        if order == 0:
            self.vmin_6 = min(s23)
            self.vmax_6 = max(s23)
        ax6 = fig.add_subplot(236, projection='3d')
        ax6.view_init(30, 30)
        plt.gca().set_box_aspect((8, 30, 25))
        p6 = ax6.scatter(z_coord, x_coord, y_coord, s=2, cmap="jet", c=s12, label="Predicted " + namestring + " (MPa)")
        p6.set_clim(self.vmin_6, self.vmax_6)
        fig.colorbar(p6, fraction=0.045, pad=0.05)
        ax6.set_xlabel('X', fontsize=10)
        ax6.set_ylabel('Y', fontsize=10)
        ax6.set_zlabel('Z', fontsize=10)
        plt.title("Predicted Stress " + namestring)

        plt.legend()

        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        buf_ndarray = np.frombuffer(fig.canvas.tostring_rgb(), dtype='u1')
        tmp_img = buf_ndarray.reshape(h, w, 3)
        cv2.imshow("STRESS", tmp_img)
        cv2.waitKey(5)
        cv2.imwrite(str(order) + ".bmp", tmp_img)
        plt.close(fig)


'''To evaluation all layers and demonstrate real-time prediction'''
def eval_all_layers(list_load, save=False):

    ratio_train = True
    '''COORD'''

    '''BM'''
    vect_length_BM = 15184
    Code_predictor_BM = Predictor(code_length=72, params_length=3, hiden_length=72).cuda()
    checkpoint = torch.load("BM_Predictor_best.pth")
    Code_predictor_BM.load_state_dict(checkpoint)
    Code_predictor_BM.eval()
    autocoder_BM = blocked_AUTOCODER(vect_length=vect_length_BM, code_length=72, num_layer=3, in_channel=6).cuda()
    autocoder_BM.load_state_dict(torch.load("BM_coder_best.pth"))
    autocoder_BM.eval()

    '''NB'''
    vect_length_NB = 13120
    Code_predictor_NB = Predictor(code_length=72, params_length=3, hiden_length=72).cuda()
    checkpoint = torch.load("NB_Predictor_best.pth")
    Code_predictor_NB.load_state_dict(checkpoint)
    Code_predictor_NB.eval()
    autocoder_NB = blocked_AUTOCODER(vect_length=vect_length_NB, code_length=72, num_layer=3, in_channel=6).cuda()
    autocoder_NB.load_state_dict(torch.load("NB_coder_best.pth"))
    autocoder_NB.eval()

    '''PR'''
    vect_length_PR = 26016
    Code_predictor_PR = Predictor(code_length=72, params_length=3, hiden_length=72).cuda()
    checkpoint = torch.load("PR_Predictor_best.pth")
    Code_predictor_PR.load_state_dict(checkpoint)
    Code_predictor_PR.eval()
    autocoder_PR = blocked_AUTOCODER_pr(vect_length=vect_length_PR, code_length=72, num_layer=3, in_channel=6).cuda()
    autocoder_PR.load_state_dict(torch.load("PR_coder_best.pth"))
    autocoder_PR.eval()
    ratio_vect = [4.5, 1.0, 4.5, 12.0, 25.0, 8.0]

    my_ploter = plot_contour()
    my_coord_predictor = coord_predictor()

    for i in range(len(list_load)):
        now = time.perf_counter()
        temp_input = torch.from_numpy(np.array(list_load[i], dtype="float32")).unsqueeze(0).cuda(0)
        '''COORD'''
        vect_outputs_bm, vect_outputs_pr, vect_outputs_nb = my_coord_predictor.runcase(temp_input)
        coord_vect_all = np.concatenate([vect_outputs_bm, vect_outputs_pr, vect_outputs_nb], axis=1)
        '''STRESS'''
        stress_outputs_BM = autocoder_BM(Code_predictor_BM(temp_input).unsqueeze(1), coder=1)[0, :].detach().cpu().numpy().T / np.array(ratio_vect)
        stress_outputs_PR = autocoder_PR(Code_predictor_PR(temp_input).unsqueeze(1), coder=1)[0, :].detach().cpu().numpy().T / np.array(ratio_vect)
        stress_outputs_NB = autocoder_NB(Code_predictor_NB(temp_input).unsqueeze(1), coder=1)[0, :].detach().cpu().numpy().T / np.array(ratio_vect)

        stress_outputs_all = np.concatenate([stress_outputs_BM, stress_outputs_PR, stress_outputs_NB], axis=0).T
        print("Time Consumption on Forward Propegation", time.perf_counter() - now)
        now = time.perf_counter()
        my_ploter.refresh(vect_outputs=stress_outputs_all, coord_vect=coord_vect_all, inputs_num=np.array(list_load[i], dtype="float32"), order=i)
        print("Time Consumption on Plot", time.perf_counter() - now)


if __name__ == '__main__':
    arr = np.loadtxt(r"I:\DigitRubber_Dataset\NPY\VALID_RESULTS\\seriel2.txt", delimiter="\t", dtype="float32")
    arr = arr[:, [1, 2, 0]]
    eval_all_layers(arr.tolist())