"""
Quick fix script to add time dimension to metric tensors for WarpFactory compatibility
"""
import sys
import re

def fix_get_metric_tensor_function(filename):
    """Add time dimension reshaping to get_metric_tensor functions"""

    with open(filename, 'r') as f:
        content = f.read()

    # Pattern to find: metric_dict = three_plus_one_builder(alpha, beta, gamma)
    # We need to add reshaping before this line

    # Find the get_metric_tensor or get_metric_3plus1 function
    pattern = r'(def get_metric_tensor.*?)(metric_dict = three_plus_one_builder\(alpha, beta, gamma\))'

    replacement_code = r'''\1# Add time dimension to make 4D arrays (required by WarpFactory)
        alpha_4d = alpha[np.newaxis, :, :, :]
        beta_4d = {key: val[np.newaxis, :, :, :] for key, val in beta.items()}
        gamma_4d = {key: val[np.newaxis, :, :, :] for key, val in gamma.items()}

        \2'''

    # Replace with version that adds time dimension
    new_content = re.sub(pattern, replacement_code, content, flags=re.DOTALL)

    # Now fix the three_plus_one_builder call
    new_content = new_content.replace(
        'metric_dict = three_plus_one_builder(alpha, beta, gamma)',
        'metric_dict = three_plus_one_builder(alpha_4d, beta_4d, gamma_4d)'
    )

    with open(filename, 'w') as f:
        f.write(new_content)

    print(f"Fixed {filename}")

if __name__ == "__main__":
    import glob
    approach_files = glob.glob('approach*.py')
    for f in approach_files:
        try:
            fix_get_metric_tensor_function(f)
        except Exception as e:
            print(f"Error fixing {f}: {e}")
