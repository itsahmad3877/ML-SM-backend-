csv_filename = 'social_media_user_behavior.csv'

if not os.path.exists(csv_filename):
    raise FileNotFoundError(f"Ensure '{csv_filename}' is uploaded directly to your Colab workspace root directory.")

# Load raw attributes
df = pd.read_csv(csv_filename)
print(f" -> Initial shape loaded: {df.shape}")

# Drop operational indices to prevent artificial tracking biases
df = df.drop(columns=['user_id', 'account_join_date'], errors='ignore')

# Deduplicate identical value streams
df.drop_duplicates(inplace=True)

# Standard Data Imputation Rule matching deployment configurations
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = df[col].fillna(df[col].mode()[0])
    else:
        df[col] = df[col].fillna(df[col].median())

# Statistical Feature Construction (Phase 4)
df['engagement_density_ratio'] = df['likes_given_per_day'] / (df['posts_per_week'] + 1)
df['session_intensity'] = df['avg_session_duration_min'] * df['sessions_per_day']

print(f" -> Processing completed. New dataset matrix shape: {df.shape}")