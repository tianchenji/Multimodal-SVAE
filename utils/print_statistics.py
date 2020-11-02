import numpy as np
import pandas as pd

def print_statistics(correct, confusion_m, total, confusion_m_flag):

    if confusion_m_flag == 0:
        accuracy = 100 * np.array(correct) / np.array(total)
        index = ['normal', 'untvbl obs', 'tvbl obs', 'crash']
        columns = ['accuracy']
        print('Accuracy of the network on the test set:')
        print(pd.DataFrame(accuracy, index, columns).round(2))

        pe_rows = np.sum(confusion_m, axis=0)
        pe_cols = np.sum(confusion_m, axis=1)
        sum_total = sum(pe_cols)
        pe = np.dot(pe_rows, pe_cols) / float(sum_total**2)
        po = np.trace(confusion_m) / float(sum_total)
        kappa = (po - pe) / (1 - pe)
        print('Kappa coefficient on the test set: {:.2f}'.format(kappa))

    else:
        confusion_m = 100 * np.array(confusion_m) / np.array(total)
        index = [['', 'predicted', 'class', ''], ['normal', 'untvbl obs', 'tvbl obs', 'crash']]
        columns = [['', 'actual class', '', ''], ['normal', 'untvbl obs', 'tvbl obs', 'crash']]
        print('Confusion matrix on the test set:')
        print(pd.DataFrame(confusion_m, index, columns).round(2))