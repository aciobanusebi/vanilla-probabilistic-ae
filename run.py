import os

OUTPUT_DIRECTORY = "../results"

for number_of_distributions in [1,10]:
  for arch_type in ["mlp","cnn"]:
    for dist_type in ["mse", "normal_without_cov", "normal_with_diag_cov", "normal", "t_without_cov", "t_with_diag_cov", "t", "nf", "matrix_mse", "matrix_normal_without_cov", "matrix_normal_with_diag_cov", "matrix_normal", "matrix_t_without_cov", "matrix_t_with_diag_cov", "matrix_t", "vae"]:
      for dataset_name in ["mnist", "fashion_mnist"]:
        command = "python ./main.py " + " " + OUTPUT_DIRECTORY + " " + str(number_of_distributions) + " " + arch_type + " " + dist_type + " " + dataset_name
        print(command)
        os.system(command)