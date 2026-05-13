import pandas as pd

df = pd.read_csv(r"C:\Users\adith\github repos\ds301scratch\data\annotated\test_df_case_level_annotated.csv")
df = df[df["label"] != -1].copy()
df = df.dropna(subset=["gpt_pet_hostility", "gpt_res_hostility"])

corr = df["hostility_diff"].corr(df["label"])

if corr > 0:
    df["predicted"] = (df["hostility_diff"] > 0).astype(int)
else:
    df["predicted"] = (df["hostility_diff"] < 0).astype(int)
correct = (df["predicted"] == df["label"]).sum()
total = len(df)

print(f"Correct: {correct} / {total} = {correct/total:.3f}")
print(f"Majority baseline: {df['label'].value_counts(normalize=True).max():.3f}")

pet_wins = df[df["label"] == 1]
res_wins = df[df["label"] == 0]
print(f"\nPetitioner wins correctly predicted: {(pet_wins['predicted']==1).sum()} / {len(pet_wins)}")
print(f"Respondent wins correctly predicted: {(res_wins['predicted']==0).sum()} / {len(res_wins)}")