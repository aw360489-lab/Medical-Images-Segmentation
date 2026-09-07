import nibabel as nib
import matplotlib.pyplot as plt


img_id = "00495"
plt.figure(figsize=(18, 5))
for i, nii in enumerate([f'extract/BraTS2021_{img_id}/BraTS2021_{img_id}_{s_type}.nii.gz' for s_type in ["flair", "t1", "t1ce", "t2", "seg"]]):
    plt.subplot(1,5,i+1)
    image=nib.load(nii).get_fdata()
    plt.title(nii.rsplit("_", 1)[1].split(".", 1)[0], fontweight="bold")
    plt.axis(False)
    plt.imshow(image[:, :, 80], cmap="gray")
plt.tight_layout()
plt.show()

img_id = "00621"
plt.figure(figsize=(18, 5))
for i, nii in enumerate([f'extract/BraTS2021_{img_id}/BraTS2021_{img_id}_{s_type}.nii.gz' for s_type in ["flair", "t1", "t1ce", "t2", "seg"]]):
    plt.subplot(1,5,i+1)
    image=nib.load(nii).get_fdata()
    plt.title(nii.rsplit("_", 1)[1].split(".", 1)[0], fontweight="bold")
    plt.axis(False)
    plt.imshow(image[:, :, 80], cmap="gray")
plt.tight_layout()
plt.show()