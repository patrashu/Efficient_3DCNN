from eval_ucf101 import UCFclassification
from eval_kinetics import KINETICSclassification



# ucf_classification = UCFclassification('../annotation_UCF101/ucf101_01.json',
#                                        '../results/val.json',
#                                        subset='validation', top_k=1)
# ucf_classification.evaluate()
# print(ucf_classification.hit_at_k)


kinetics_classification = KINETICSclassification('./hongkong/hongkong_annot.json',
                                       './results/val.json',
                                       subset='validation',
                                       top_k=5,
                                       check_status=False)
kinetics_classification.evaluate()
print(kinetics_classification.hit_at_k)
