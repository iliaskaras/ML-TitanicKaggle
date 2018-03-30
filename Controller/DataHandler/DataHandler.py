import numpy as np


class DataHandler(object):

    def changeNonNumericalData(self, df):
        columns = df.columns.values

        for column in columns:
            text_digit_vals = {}

            def convertToInt(val):
                return text_digit_vals[val]

            if df[column].dtype != np.int64 and df[column].dtype != np.float64:
                column_contents = df[column].values.tolist()
                unique_elements = set(column_contents)
                x = 0
                for unique in unique_elements:
                    if unique not in text_digit_vals:
                        text_digit_vals[unique] = x
                        x += 1

                df[column] = list(map(convertToInt, df[column]))

        return df
