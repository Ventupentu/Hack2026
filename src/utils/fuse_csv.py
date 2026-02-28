import pandas as pd

df1 = pd.read_csv("/home/sgrodriguez23/Hack2026/submission_part_0.csv")
df2 = pd.read_csv("/home/sgrodriguez23/Hack2026/submission_part_1.csv")

df_final = pd.concat([df1, df2], ignore_index=True)
df_final.to_csv("/home/sgrodriguez23/Hack2026/submission_final.csv", index=False)
print("¡Submission unificada y lista para enviar!")