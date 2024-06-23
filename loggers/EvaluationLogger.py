import csv


class EvaluationLogger:
    def __init__(self, file_path):
        self.file_path = file_path
        self.initialize_csv()

    def initialize_csv(self):
        with open(self.file_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Type', 'Data'])

    def log_mAP_detection(self, mAP, data_split):
        with open(self.file_path, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([f'mAP {data_split}', mAP])
        print(f'mAP detection saved to {self.file_path}')

    def log_mAP_classification(self, ap_per_class, mAP, data_split):
        with open(self.file_path, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([f'classification AP per class {data_split}', ap_per_class])
            writer.writerow([f'classification mAP {data_split}', mAP])
        print(f'mAP classification saved to {self.file_path}')

    def log_mAP_pin_pointing(self, ap_per_class, mAP, data_split):
        with open(self.file_path, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([f'pin pointing AP per class {data_split}', ap_per_class])
            writer.writerow([f'pin pointing mAP {data_split}', mAP])
        print(f'mAP pin pointing saved to {self.file_path}')

    def log_cor_loc(self, cor_loc_per_class, mean_cor_loc, data_split):
        with open(self.file_path, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([f'CorLoc per class {data_split}', cor_loc_per_class])
            writer.writerow([f'mean CorLoc {data_split}', mean_cor_loc])
        print(f'CorLoc saved to {self.file_path}')

    def log_time(self, time):
        with open(self.file_path, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([f'Total inference time for whole dataset', time])
        print(f'Evaluation time saved to {self.file_path}')
