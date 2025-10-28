# 📚 Import Libraries
import pandas as pd
import matplotlib.pyplot as plt

# 🧠 Create Sample DataFrame
data = {
    'Name': ['Raj', 'Neha', 'Amit', 'Priya', 'Karan'],
    'Age': [25, None, 30, 28, None],
    'Salary': [30000, 28000, None, 35000, 40000]
}

df = pd.DataFrame(data)
print("📊 Original DataFrame:\n", df)

# 🔍 Check Missing Values Before Cleaning
missing_before = df.isnull().sum()
print("\n🔹 Missing Values (Before Cleaning):\n", missing_before)

# 🧹 Fill Missing Values
df = df.fillna({
    'Age': df['Age'].mean(),
    'Salary': df['Salary'].median()
})

# 🔍 Check Missing Values After Cleaning
missing_after = df.isnull().sum()
print("\n✅ Cleaned DataFrame:\n", df)

# 📈 Compare Missing Values Before vs After
plt.figure(figsize=(7,4))
plt.bar(missing_before.index, missing_before.values, label='Before Cleaning', alpha=0.7)
plt.bar(missing_after.index, missing_after.values, label='After Cleaning', alpha=0.7)
plt.title("🧩 Missing Values Before vs After Cleaning", fontsize=12)
plt.ylabel("Count of Missing Values")
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.show()
