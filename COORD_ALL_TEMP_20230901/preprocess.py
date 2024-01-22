import numpy as np
import xlrd
import os
import re
import shutil


def read_xlsx(file_name):
    data = xlrd.open_workbook(file_name)
    table = data.sheets()[0]
    x_vect = []
    y_vect = []
    z_vect = []
    all_vect = []
    for row in range(table.nrows):
        x_vect.append(table.cell_value(row, 4))
        y_vect.append(table.cell_value(row, 5))
        z_vect.append(table.cell_value(row, 6))
        all_vect.append(table.cell_value(row, 4))
        all_vect.append(table.cell_value(row, 5))
        all_vect.append(table.cell_value(row, 6))
    coord = np.array([x_vect, y_vect, z_vect], dtype="float32")
    all_array = np.array(all_vect, dtype="float32")
    return coord, all_array


def read_xlsx_stress(file_name):
    data = xlrd.open_workbook(file_name)
    table = data.sheets()[1]
    S11_vect = []
    S22_vect = []
    S33_vect = []
    S12_vect = []
    S13_vect = []
    S23_vect = []
    all_vect = []
    for row in range(table.nrows):
        S11_vect.append(table.cell_value(row, 1))
        S22_vect.append(table.cell_value(row, 2))
        S33_vect.append(table.cell_value(row, 3))
        S12_vect.append(table.cell_value(row, 4))
        S13_vect.append(table.cell_value(row, 5))
        S23_vect.append(table.cell_value(row, 6))
        # all_vect.append(table.cell_value(row, 1))
        # all_vect.append(table.cell_value(row, 2))
        # all_vect.append(table.cell_value(row, 3))
        # all_vect.append(table.cell_value(row, 4))
        # all_vect.append(table.cell_value(row, 5))
        # all_vect.append(table.cell_value(row, 6))

    all_array = np.array([S11_vect, S22_vect, S33_vect, S12_vect, S13_vect, S23_vect], dtype="float32")
    # all_vect = np.array(all_vect, dtype="float32")
    return all_array#, all_vecdt


def read_xlsx_FD(file_name, max_pressure):

    data = xlrd.open_workbook(file_name)
    table = data.sheets()[0]
    disp = []
    load = []
    pressure = []
    disp_0 = []
    load_0 = []
    pressure_0 = []
    # if max_pressure == 0:
    for row in range(table.nrows-1):
        if table.cell_value(row+1, 0) - table.cell_value(row, 0) > 0.0001:
            disp_0.append(table.cell_value(row, 0))
            pressure_0.append(0)
            load_0.append(table.cell_value(row, 1))
        else:
            pass
    # disp_0.append(table.cell_value(row+1, 0))
    # pressure_0.append(max_pressure)
    # load_0.append(table.cell_value(row+1, 1))

    # else:
    disp.append(table.cell_value(table.nrows-1, 0))
    pressure.append(max_pressure)
    load.append(table.cell_value(table.nrows-1, 1))

    return np.array([disp_0, pressure_0, load_0]), np.array([disp, pressure, load])


def generate_FD(xlsx_dir, savedir):
    allfiles = os.listdir(xlsx_dir)
    names = [file for file in allfiles if file.endswith("-st2.xlsx") and file.startswith("FD-")]
    filenames = [os.path.join(xlsx_dir, file) for file in names]
    for i in range(len(filenames)):
        split_keywords = names[i].split("-")[:-1]
        temperature = int(split_keywords[1][:-3])
        delta_pressure = 100 - int(split_keywords[3][:-3])
        displacement = float(split_keywords[4][:-2])
        temp_fname = filenames[i]
        temp_array_0, temp_array = read_xlsx_FD(temp_fname, max_pressure=delta_pressure)
        savename_0 = savedir + "FD_" + str(temperature) + "_" + str(0) + "_" + str(displacement) + "_FD.npy"
        savename = savedir + "FD_" + str(temperature) + "_" + str(delta_pressure) + "_" + str(displacement) + "_FD.npy"
        # if np.random.rand() > 0.8:
        #     savename = savedir + "TRAIN/" + "FD_" + str(temperature) + "_" + str(delta_pressure) + "_" + str(
        #         displacement) + "_FD.npy"
        # else:
        #     savename = savedir + "VALID/" + "FD_" + str(temperature) + "_" + str(delta_pressure) + "_" + str(
        #         displacement) + "_FD.npy"
        np.save(savename, temp_array)
        np.save(savename_0, temp_array_0)


def generate_dataset(xlsx_dir, save_dir, train_percent=0.8, kw="BM"):

    allfiles = os.listdir(xlsx_dir)
    allfiles = [file for file in allfiles if file.endswith("STRESS.xlsx") and file.startswith(kw)]
    file_dirs = [os.path.join(xlsx_dir, file) for file in allfiles]
    np.random.seed(1)

    for i in range(len(file_dirs)):
        temp_filename = file_dirs[i]
        split_keywords = allfiles[i].split("-")[1:-1]
        type = kw
        # temperature = int(split_keywords[1][:-3])
        # delta_pressure = 100 - int(split_keywords[3][:-3])
        # displacement = float(split_keywords[4][:-2])
        # st = float(split_keywords[5][-1])
        print(temp_filename)
        if (split_keywords[0] == ''):
            stt = 1
        else:
            stt = 0
        temperature = -int(split_keywords[stt+0][:-3])
        delta_pressure = 100 - int(split_keywords[stt + 2][:-3])
        displacement = float(split_keywords[stt + 3][:-2])
        st = float(split_keywords[stt + 4][2:])



        if st <= 1:
            delta_pressure = 0
            # ratio = float(split_keywords[6])
            # ratio = (ratio < 1) * (ratio - 1) + 1
            # real_displecement = round(ratio * displacement, 2)
            ratio = st
            real_displecement = round(ratio * displacement, 3)
            print(ratio, displacement)
        else:
            real_displecement = displacement


        if np.random.rand() < train_percent:
            temp_save_name = save_dir + "TRAIN/" + type + "_" + str(temperature) + "_" + str(delta_pressure) + "_" + str(real_displecement) + "_COORD.npy"
        else:
            temp_save_name = save_dir + "VALID/" + type + "_" + str(temperature) + "_" + str(delta_pressure) + "_" + str(real_displecement) + "_COORD.npy"

        if os.path.exists(temp_save_name):
            continue


        # if np.random.rand() < train_percent:
        #     temp_save_name = save_dir + "TRAIN/" + type + "_" + str(temperature) + "_" + str(delta_pressure) + "_" + str(real_displecement) + "_Coord.npy"
        # else:
        #     temp_save_name = save_dir + "VALID/" + type + "_" + str(temperature) + "_" + str(delta_pressure) + "_" + str(real_displecement) + "_Coord.npy"
        # temp_save_name = save_dir + type + "_" + str(temperature) + "_" + str(delta_pressure) + "_" + str(
        #     real_displecement) + "_Coord.npy"
        #
        # np.save(temp_save_name, coord_vect)
        #
        # print("\rFinish %d of %d "%(i, len(allfiles)), end="\b")

        print(temp_filename)

        coord_array, coord_vect = read_xlsx(temp_filename)
        np.save(temp_save_name, coord_array)
        print("\rFinish %d of %d "%(i, len(allfiles)), end="\b")


def generate_dataset_STRESS(xlsx_dir, save_dir, train_percent=0.8, kw="BM"):

    allfiles = os.listdir(xlsx_dir)
    allfiles = [file for file in allfiles if file.endswith("STRESS.xlsx") and file.startswith(kw)]
    file_dirs = [os.path.join(xlsx_dir, file) for file in allfiles]
    np.random.seed(1)

    for i in range(len(file_dirs)):
        temp_filename = file_dirs[i]
        split_keywords = allfiles[i].split("-")[1:-1]
        type = kw
        temperature = -int(split_keywords[1][:-3])
        delta_pressure = 100 - int(split_keywords[3][:-3])
        displacement = float(split_keywords[4][:-2])
        st = float(split_keywords[5][2:])
        if st <= 1:
            delta_pressure = 0
            # ratio = float(split_keywords[6])
            ratio = st
            # ratio = (ratio < 1) * (ratio - 1) + 1
            real_displecement = round(ratio * displacement, 3)
            print(ratio, displacement)
        else:
            real_displecement = displacement

        stress_array = read_xlsx_stress(temp_filename)

        if np.random.rand() < train_percent:
            temp_save_name = save_dir + "TRAIN/" + type + "_" + str(temperature) + "_" + str(delta_pressure) + "_" + str(real_displecement) + "_STRESS.npy"
        else:
            temp_save_name = save_dir + "VALID/" + type + "_" + str(temperature) + "_" + str(delta_pressure) + "_" + str(real_displecement) + "_STRESS.npy"
        # temp_save_name = save_dir + type + "_" + str(temperature) + "_" + str(delta_pressure) + "_" + str(
        #     real_displecement) + "_STRESS.npy"
        print(temp_save_name)
        np.save(temp_save_name, stress_array)

        print("\rFinish %d of %d "%(i, len(allfiles)), end="\b")


def reassign_dataset(dataset_dir, reference_dir, new_dir):
    train_dir = reference_dir + "TRAIN\\"
    test_dir = reference_dir + "VALID\\"
    train_files = [file for file in os.listdir(train_dir) if file.endswith("STRESS.npy")]
    test_files = [file for file in os.listdir(test_dir) if file.endswith("STRESS.npy")]
    all_train_files = [file for file in os.listdir(dataset_dir + "TRAIN\\") if file.endswith("Coord.npy")]
    all_test_files = [file for file in os.listdir(dataset_dir + "VALID\\") if file.endswith("Coord.npy")]

    for i in range(len(all_train_files)):

        temp_name = all_train_files[i][:-9]+"STRESS.npy"
        if temp_name in train_files:
            temp_save_name = new_dir + "TRAIN\\" + all_train_files[i]
        elif temp_name in test_files:
            temp_save_name = new_dir + "VALID\\" + all_train_files[i]
        else:
            exit(-100)

        shutil.copyfile(dataset_dir + "TRAIN\\" + all_train_files[i], temp_save_name)
        print("copy "+dataset_dir + "TRAIN\\" + all_train_files[i] + "\t to \t "+temp_save_name)
    print(i)

    for i in range(len(all_test_files)):

        temp_name = all_test_files[i][:-9]+"STRESS.npy"
        if temp_name in train_files:
            temp_save_name = new_dir + "TRAIN\\" + all_test_files[i]
        elif temp_name in test_files:
            temp_save_name = new_dir + "VALID\\" + all_test_files[i]
        else:
            exit(-100)
        shutil.copyfile(dataset_dir + "VALID\\" +all_test_files[i], temp_save_name)
        print("copy "+dataset_dir + "TRAIN\\" +all_test_files[i] + "\t to \t "+temp_save_name)
    print(i)


if __name__ == '__main__':
    # generate_dataset_STRESS(xlsx_dir=r"E:\Data\DATASET\SealDigitTwin\TransLearn\20221005LT\\",
    #                         save_dir=r"E:\Data\DATASET\SealDigitTwin\TransLearn\NPY\\", kw="BM")
    # generate_dataset_STRESS(xlsx_dir=r"E:\Data\DATASET\SealDigitTwin\TransLearn\20221005LT\\",
    #                         save_dir="E:\Data\DATASET\SealDigitTwin\Results\\0_input_npy\\STRESS\\", kw="NB")
    # generate_dataset_STRESS(xlsx_dir="E:\Data\DATASET\SealDigitTwin\ADD\\",
    #                         save_dir="E:\Data\DATASET\SealDigitTwin\Results\\0_input_npy\\STRESS\\", kw="PR")
    generate_dataset(xlsx_dir=r"E:\Data\DATASET\SealDigitTwin\TransLearn\20221005LT\\",
                     save_dir=r"E:\Data\DATASET\SealDigitTwin\TransLearn\NPY\\", kw="BM")
    generate_dataset(xlsx_dir=r"E:\Data\DATASET\SealDigitTwin\TransLearn\20221005LT\\",
                     save_dir=r"E:\Data\DATASET\SealDigitTwin\TransLearn\NPY\\", kw="NB")
    generate_dataset(xlsx_dir=r"E:\Data\DATASET\SealDigitTwin\TransLearn\20221005LT\\",
                     save_dir=r"E:\Data\DATASET\SealDigitTwin\TransLearn\NPY\\", kw="PR")

    # generate_dataset(xlsx_dir="E:\Data\DATASET\SealDigitTwin\ADD\\", save_dir="E:\Data\DATASET\SealDigitTwin\Results\\0_input_npy\\COORD\\", kw="BM")
    # generate_dataset(xlsx_dir="E:\Data\DATASET\SealDigitTwin\ADD\\", save_dir="E:\Data\DATASET\SealDigitTwin\Results\\0_input_npy\\COORD\\", kw="NB")
    # generate_dataset(xlsx_dir="E:\Data\DATASET\SealDigitTwin\ADD\\", save_dir="E:\Data\DATASET\SealDigitTwin\Results\\0_input_npy\\COORD\\", kw="PR")


    # generate_dataset(xlsx_dir="I:\DigitRubber_Dataset\INIT\\", save_dir="I:\DigitRubber_Dataset\\NPY2\\COORD\\", kw="NB")
    # generate_dataset_STRESS(xlsx_dir="I:\DigitRubber_Dataset\INIT\\", save_dir="I:\DigitRubber_Dataset\\NPY2\\STRESS\\", kw="NB")
    # generate_dataset(xlsx_dir="I:\DigitRubber_Dataset\INIT\\", save_dir="I:\DigitRubber_Dataset\\NPY2\\COORD\\", kw="PR")
    # generate_dataset_STRESS(xlsx_dir="I:\DigitRubber_Dataset\INIT\\", save_dir="I:\DigitRubber_Dataset\\NPY2\\STRESS\\", kw="PR")
    # reassign_dataset(dataset_dir="I:\\DigitRubber_Dataset\\NPY\\COORD\\",
    #                  reference_dir="I:\\DigitRubber_Dataset\\NPY\\STRESS\\",
    #                  new_dir="I:\\DigitRubber_Dataset\\NPY\\COORD1\\")
    # rearrange_dataset()
    # generate_FD(xlsx_dir=r"E:\Data\DATASET\SealDigitTwin\STRESS\LOAD\FD0130\\", savedir=r"E:\Data\DATASET\SealDigitTwin\STRESS\LOAD\NPY\\")
    # a = []


