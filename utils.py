from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
import rasterio
from rasterio.windows import from_bounds
import matplotlib.pyplot as plt
import rasterio
from skimage.filters import threshold_otsu
import numpy as np
import matplotlib.patches as mpatches
from matplotlib.patches import Patch
import joblib
import pandas as pd
from scipy.ndimage import uniform_filter
from sklearn.decomposition import PCA
import geopandas as gpd
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
import matplotlib.colors as mcolors
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from skimage.morphology import remove_small_objects
from sklearn.metrics import log_loss, roc_auc_score, roc_curve
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay,classification_report

def evaluate_model(y_true, y_pred, average='weighted'):
    """
    Compute and return classification evaluation metrics.
    
    Parameters:
        y_true (array-like): True labels.
        y_pred (array-like): Predicted labels.
        average (str): Averaging method for multi-class metrics. Default is 'weighted'.
        
    Returns:
        dict: Dictionary containing accuracy, f1-score, precision, recall, and roc-auc-score.
    """

    cm = confusion_matrix(y_true, y_pred)
    metrics = {
        "log_loss": log_loss(y_true, y_pred),
        'accuracy': accuracy_score(y_true, y_pred),
        'f1_score': f1_score(y_true, y_pred, average=average),
        'precision': precision_score(y_true, y_pred, average=average, zero_division=0),
        'recall': recall_score(y_true, y_pred, average=average),
        # 'roc_auc': roc_auc_score(y_true, y_pred, multi_class='ovr') if len(set(y_true)) > 2 else roc_auc_score(y_true, y_pred)
        'roc_auc': roc_auc_score(y_true, y_pred),
         "confusion_matrix": cm
    }
    
    # Display it
    # disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    # disp.plot(cmap=plt.cm.Blues)
    # plt.title("Confusion Matrix")
    # plt.show()

    print('Model performance')
    # print(log_loss(y_true, y_pred))
    print(f"- Log loss: {metrics['log_loss']}")
    print(f"- Accuracy: {metrics['accuracy']}")
    print(f"- F1 Score: {metrics['f1_score']}")
    print(f"- Precision Score: {metrics['precision']}")
    print(f"- Recall Score: {metrics['accuracy']}")
    print(f"- Roc Auc Score: {metrics['roc_auc']}")
    print(f"- confusion_matrix: {metrics['confusion_matrix']}")

    return metrics



def clip_image(input_path, output_path, clip_extent):
    try:
        with rasterio.open(input_path) as dataset:
            window = from_bounds(*clip_extent, dataset.transform)
            clipped_transform = dataset.window_transform(window)
            clipped_data = dataset.read(1, window=window)
            clipped_meta = dataset.meta.copy()
            
            count=dataset.count
            clipped_meta.update({
            'height': clipped_data.shape[0],
            'width': clipped_data.shape[1],
            'transform': clipped_transform
            })
            with rasterio.open(output_path, 'w', **clipped_meta) as dest:
                for i in range(count):
                    clipped_data = dataset.read(i+1, window=window)
                    dest.write(clipped_data, i+1)  # Write the first band
        print(f"clip image saved into  {output_path}")
    except Exception as e:
        print("clip_image error: ",str(e))


def sar_single_date_image(input_sar_image_path,output_sar_image_path):
    try:
        with rasterio.open(input_sar_image_path) as dataset:
            vh_pre_3= dataset.read(5)
            vv_pre_3= dataset.read(6)

            vh_post_1= dataset.read(7)
            vv_post_1= dataset.read(8)

            meta=dataset.meta.copy()
            meta.update({"count":4})

        with rasterio.open(output_sar_image_path, 'w', **meta) as dest:
            dest.write(vh_pre_3, 1)
            dest.write(vv_pre_3, 2)
            dest.write(vh_post_1, 3)
            dest.write(vv_post_1, 4)
        print(f"Average image saved into  {output_sar_image_path}")
    except Exception as e:
        print("sar average error: ", str(e))

def sar_image_average(input_sar_image_path,output_sar_image_path):
    try:
        with rasterio.open(input_sar_image_path) as dataset:
            # pre_vh_bands = dataset.read()
            vh_pre_1 = dataset.read(1)
            vv_pre_1 = dataset.read(2)

            vh_pre_2 = dataset.read(3)
            vv_pre_2 = dataset.read(4)

            vh_pre_3= dataset.read(5)
            vv_pre_3= dataset.read(6)

            vh_post_1= dataset.read(7)
            vv_post_1= dataset.read(8)

            vh_post_2= dataset.read(9)
            vv_post_2= dataset.read(10)

            vh_post_3= dataset.read(11)
            vv_post_3= dataset.read(12)
        
            meta=dataset.meta.copy()
            meta.update({"count":4})

        with rasterio.open(output_sar_image_path, 'w', **meta) as dest:
            
            vh_pre = (vh_pre_1+vh_pre_2 +vh_pre_3)/3
            vv_pre = (vv_pre_1+vv_pre_2 + vv_pre_3)/3
            # vh_pre = (vh_pre_2 +vh_pre_3)/2
            # vv_pre = (vv_pre_2 + vv_pre_3)/2
            vh_post= (vh_post_1+vh_post_2+vh_post_3)/3
            vv_post= (vv_post_1+vv_post_2+vv_post_3)/3
            dest.write(vh_pre, 1)
            dest.write(vv_pre, 2)
            dest.write(vh_post, 3)
            dest.write(vv_post, 4)
        print(f"Average image saved into  {output_sar_image_path}")
    except Exception as e:
        print("sar average error: ", str(e))

# def merge_asc_desc_sar(asc_image_path, desc_image_path, output_path):
#     try:
#         with rasterio.open(asc_image_path) as asc_dataset, rasterio.open(desc_image_path) as desc_dataset:
#             meta=asc_dataset.meta.copy()
#             asc_vh_pre = asc_dataset.read(1)
#             asc_vv_pre = asc_dataset.read(2)

#             asc_vh_post = asc_dataset.read(3)
#             asc_vv_post = asc_dataset.read(4)

        #     desc_vh_pre = desc_dataset.read(1)
        #     desc_vv_pre = desc_dataset.read(2)

        #     desc_vh_post = desc_dataset.read(3)
        #     desc_vv_post = desc_dataset.read(4)

        # with rasterio.open(output_path, 'w', **meta) as dest:
        #     vh_pre = (asc_vh_pre+desc_vh_pre )/2
        #     vv_pre = (asc_vv_pre+desc_vv_pre )/2
        #     vh_post= (asc_vh_post+desc_vh_post)/2
        #     vv_post= (asc_vv_post+desc_vv_post)/2
        #     # vh_pre = np.maximum(asc_vh_pre,desc_vh_pre )
        #     # vv_pre = np.maximum(asc_vv_pre,desc_vv_pre )
        #     # vh_post= np.maximum(asc_vh_post,desc_vh_post)
        #     # vv_post= np.maximum(asc_vv_post,desc_vv_post)
        #     dest.write(vh_pre, 1)
        #     dest.write(vv_pre, 2)
        #     dest.write(vh_post, 3)
        #     dest.write(vv_post, 4)
        # print(f"asc_desc image saved into  {output_path}")


    except Exception as e:
        print("error in merge asc desc: ", str(e))


def generate_gt(dnbr_file_path, output_gt_path):
    try:
        with rasterio.open(dnbr_file_path) as src:
            dNBR_data = src.read(1)  # Read the first band (assuming it's a single-band image)
            # print(np.min(dNBR_data),"min")
        # dNBR_data = np.nan_to_num(dNBR_data, nan=0, posinf=0, neginf=0)
        dNBR_data[dNBR_data == -3.4028235e+38] = np.nan  # Replace with NaN for better handling

        # Optionally, you can replace NaN values with 0 if you want
        dNBR_data = np.nan_to_num(dNBR_data, nan=0)
        # Flatten the data to 1D array for histogram calculation
        dNBR_flat = dNBR_data.flatten()

        # Plot histogram of dNBR values
        plt.hist(dNBR_flat, bins=50, range=(dNBR_flat.min(), dNBR_flat.max()), alpha=0.75, color='blue')
        plt.title("Histogram of dNBR Values")
        plt.xlabel("dNBR values")
        plt.ylabel("Frequency")
        plt.show()

        # Apply Otsu's thresholding method to find the optimal threshold
        threshold = threshold_otsu(dNBR_data)
        print(f"Optimal threshold based on Otsu's method: {threshold}")

        # Classify the dNBR image using the threshold (Burnt vs Non-burnt)
        classified_image = dNBR_data > threshold
        # classified_image = dNBR_data < 0.25

        # Plot the classified image
        # Define crop bounds in array (row, col) — e.g., top:bottom, left:right
        row_start, row_end = 100, 1400
        col_start, col_end = 550, 2550

        # Crop the classified image
        clipped_img = classified_image[row_start:row_end, col_start:col_end]

        # Plot the cropped image
        plt.imshow(clipped_img, cmap='gray')
        plt.title("Ground Truth")

        legend_elements = [
            mpatches.Patch(color='black', label='0 = Non-Burnt'),
            mpatches.Patch(color='white', label='1 = Burnt')
        ]
        plt.legend(handles=legend_elements, loc='upper left')

        plt.axis('off')

        # Save as PNG
        plt.savefig(f"{output_gt_path.replace('.tif','')}.png", dpi=300, bbox_inches='tight')
        plt.show()

        # Optionally, you can save the classified image as a new TIFF file
        with rasterio.open(output_gt_path, 'w', driver='GTiff', height=dNBR_data.shape[0], width=dNBR_data.shape[1],
                        count=1, dtype=dNBR_data.dtype, crs=src.crs, transform=src.transform) as dst:
            dst.write(classified_image.astype(np.uint8), 1)
        print(f"gt saved in : {output_gt_path}")
    except Exception as e:
        print("generate gt error", str(e))


"""
Log10 of negative value is 0, thus applicanle in calculating the differnce
"""
# from scipy.ndimage import generic_filter

# def extract_feature(add_bands, input_image_path, glcm_raster_path,  thermal_image_path, dnbr_image_path, final_columns_names, shp_path, is_polygon, extract_sample_feature,output_image_path, output_file_path):
def extract_feature(add_bands, input_desc_sar_image_path,input_image_path, glcm_raster_path,  thermal_image_path, dnbr_image_path, final_columns_names,output_image_path):
    try:
        epsilon = 1e-6
        is_desc_available=False
        if input_desc_sar_image_path:
            is_desc_available=True
        if input_desc_sar_image_path:
            with rasterio.open(input_desc_sar_image_path) as src:
                desc_bands = src.read()  # Read all bands

            if input_image_path:
                desc_vh_pre,desc_vv_pre,desc_vh_post, desc_vv_post=desc_bands[0],desc_bands[1],desc_bands[2],desc_bands[3]
                desc_vv_post = np.nan_to_num(desc_vv_post, nan=0)
                desc_vh_post = np.nan_to_num(desc_vh_post, nan=0)
                desc_vv_pre = np.nan_to_num(desc_vv_pre, nan=0)
                desc_vh_pre = np.nan_to_num(desc_vh_pre, nan=0)

                desc_vv_post = np.where(desc_vv_post <= 0, epsilon, desc_vv_post)
                desc_vh_post = np.where(desc_vh_post <= 0, epsilon, desc_vh_post)
                desc_vv_pre = np.where(desc_vv_pre <= 0, epsilon, desc_vv_pre)
                desc_vh_pre = np.where(desc_vh_pre <= 0, epsilon, desc_vh_pre)

            
                desc_RBD_VV_band = desc_vv_post - desc_vv_pre

                desc_RBD_VH_band = desc_vh_post - desc_vh_pre

                desc_RBR_VV_band = (10*np.log10(desc_vv_post))-(10*np.log10(desc_vv_pre)) #logarithmic ratio

                desc_RBR_VH_band = (10*np.log10(desc_vh_post)) - (10*np.log10(desc_vh_pre))
            
                desc_RVI_post = (10*np.log10((4* desc_vh_post) ))-(10*np.log10(desc_vv_post+desc_vh_post)) #4* vh/(vv+vh)
                desc_RVI_pre = (10*np.log10((4* desc_vh_pre)))-(10*np.log10(desc_vh_pre+desc_vv_pre)) #4* vh/(vv+vh)
                desc_delta_RVI = desc_RVI_post  - desc_RVI_pre
    

        # # Open input raster
        # with rasterio.open(input_image_path) as src:
        #     profile = src.profile  # Get metadata
        #     bands = src.read()  # Read all bands

        # profile.update(count=len(final_columns_names))
        if not input_image_path and glcm_raster_path:
            with rasterio.open(glcm_raster_path) as src:
                    profile = src.profile  # Get metadata
                    bands = src.read()  # Read all bands
        else:
            with rasterio.open(input_image_path) as src:
                profile = src.profile  # Get metadata
                bands = src.read()  # Read all bands
        epsilon = 1e-6
        band_index=0
        profile.update(count=len(final_columns_names))

        with rasterio.open(output_image_path, "w", **profile) as dst:
            # if add_bands:
            #     for i in range(len(bands)):
            #         dst.write(bands[i], i + 1)
            #         dst.set_band_description(i + 1, final_columns_names[i])  # Assign names
            #         band_index=i+1
            if thermal_image_path:
                with rasterio.open(thermal_image_path) as TRAD_dataset:
                    trad_band = TRAD_dataset.read()
                    band_index=band_index+1
                    dst.write(trad_band[0], band_index)
                    dst.set_band_description(band_index, "dTRAD")
            if dnbr_image_path:
                with rasterio.open(dnbr_image_path) as dnbr_dataset:
                    dnbr_band = dnbr_dataset.read()
                    band_index=band_index+1
                    dst.write(dnbr_band[0], band_index)
                    dst.set_band_description(band_index, "dNBR")
            
            if input_image_path:
                # Open input raster
                # with rasterio.open(input_image_path) as src:
                #     profile = src.profile  # Get metadata
                #     bands = src.read()  # Read all bands

                

                vh_pre,vv_pre,vh_post, vv_post=bands[0],bands[1],bands[2],bands[3]
                vv_post = np.nan_to_num(vv_post, nan=0)
                vh_post = np.nan_to_num(vh_post, nan=0)
                vv_pre = np.nan_to_num(vv_pre, nan=0)
                vh_pre = np.nan_to_num(vh_pre, nan=0)

                vv_post = np.where(vv_post <= 0, epsilon, vv_post)
                vh_post = np.where(vh_post <= 0, epsilon, vh_post)
                vv_pre = np.where(vv_pre <= 0, epsilon, vv_pre)
                vh_pre = np.where(vh_pre <= 0, epsilon, vh_pre)

                # print(np.min(vh_post),np.max(vh_post))
                band_index=band_index+1
                RBD_VV_band = vv_post - vv_pre
                if is_desc_available:
                    # RBD_VV_band=np.maximum(RBD_VV_band,desc_RBD_VV_band)
                    RBD_VV_band=(RBD_VV_band+desc_RBD_VV_band)/2

                dst.write(RBD_VV_band, band_index)
                dst.set_band_description(band_index, "RBD_VV")


                band_index=band_index+1
                RBD_VH_band = vh_post - vh_pre
                if is_desc_available:
                    # RBD_VH_band=np.maximum(RBD_VH_band,desc_RBD_VH_band)
                    RBD_VH_band=(RBD_VH_band+desc_RBD_VH_band)/2

                dst.write(RBD_VH_band, band_index)
                dst.set_band_description(band_index, "RBD_VH")

                band_index=band_index+1
                RBR_VV_band = (10*np.log10(vv_post))-(10*np.log10(vv_pre)) #logarithmic ratio
                if is_desc_available:
                    # RBR_VV_band=np.maximum(RBR_VV_band,desc_RBR_VV_band)
                    RBR_VV_band=(RBR_VV_band+desc_RBR_VV_band)/2

                dst.write(RBR_VV_band, band_index)
                dst.set_band_description(band_index, "RBR_VV")
                

                band_index=band_index+1
                RBR_VH_band = (10*np.log10(vh_post)) - (10*np.log10(vh_pre))
                if is_desc_available:
                    # RBR_VH_band=np.maximum(RBR_VH_band,desc_RBR_VH_band)
                    RBR_VH_band=(RBR_VH_band+desc_RBR_VH_band)/2

                dst.write(RBR_VH_band, band_index)
                dst.set_band_description(band_index, "RBR_VH")

                band_index=band_index+1
                # 4* vh/(vv+vh) 
                RVI_post = (10*np.log10((4* vh_post) ))-(10*np.log10(vv_post+vh_post)) #4* vh/(vv+vh)
                RVI_pre = (10*np.log10((4* vh_pre)))-(10*np.log10(vh_pre+vv_pre)) #4* vh/(vv+vh)
                delta_RVI = RVI_post  - RVI_pre
                if is_desc_available:
                    # delta_RVI=np.maximum(delta_RVI,desc_delta_RVI)
                    delta_RVI=(delta_RVI+desc_delta_RVI)/2

                dst.write(delta_RVI, band_index)
                dst.set_band_description(band_index, "ΔRVI")

            if glcm_raster_path:
                band_index=band_index+1
                with rasterio.open(glcm_raster_path) as glcm_dst:
                    glcm_bands = glcm_dst.read()  
                    for i in range(int(len(glcm_bands))):
                        dst.write(glcm_bands[i], band_index)
                        band_index=band_index+1
        print(f"feature image saved at: {output_image_path}")
    except Exception as e:
        print("feature extraction error: ", str(e))


# def create_fish_net(feature_image_path, gt_image_path, tile_size=100, plot_fig=True,train_ids=None,test_ids=None):
#     try:
#         with rasterio.open(feature_image_path) as f_src, rasterio.open(gt_image_path) as l_src:
#             features = f_src.read()  # (bands, H, W)
#             labels = l_src.read()   # (H, W)
#         tiles = []
#         positions = []
#         for row in range(0, labels.shape[0], tile_size):
#             for col in range(0, labels.shape[1], tile_size):
#                 if (row + tile_size) <= labels.shape[0] and (col + tile_size) <= labels.shape[1]:
#                     feat_patch = features[:, row:row+tile_size, col:col+tile_size]
#                     label_patch = labels[row:row+tile_size, col:col+tile_size]
#                     tiles.append(((row, col), feat_patch, label_patch))
#                     positions.append((row, col))
#         if plot_fig:
#             fig, ax = plt.subplots(figsize=(10, 10))
#             ax.imshow(labels)

#             for idx, (row, col) in enumerate(positions):
#                 if idx in train_ids:
#                     face_color = 'green'
#                 elif idx in test_ids:
#                     face_color = 'blue'
#                 # elif idx in val_ids:
#                 #     face_color = 'orange'
#                 else:
#                     # continue
#                     face_color = 'red'
#                     # opacity=1

#                 # Draw semi-transparent rectangle
#                 rect = plt.Rectangle((col, row), tile_size, tile_size,
#                                     linewidth=1.5, edgecolor=face_color,
#                                     facecolor=face_color, alpha=0.4)
#                 ax.add_patch(rect)

#                 # Label tile index
#                 ax.text(col + tile_size // 2, row + tile_size // 2, str(idx),
#                         fontsize=7, color='black', ha='center', va='center')

#             # Legend
#             legend_elements = [
#                 Patch(facecolor='green', edgecolor='green', label='Train'),
#                 Patch(facecolor='blue', edgecolor='blue', label='Test'),
#                 # Patch(facecolor='orange', edgecolor='orange', label='Validation')
#             ]
#             ax.legend(handles=legend_elements, loc='upper right')

#             plt.title("Fishnet with Colored Tiles (Transparent Fill)")
#             plt.tight_layout()
#             plt.show()
#         print("fishnet tile generation done")
#         return tiles
    
#     except Exception as e:
#         print("error in fishnet: ", str(e))
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.patches import Patch

def create_fish_net(feature_image_path, label_image_path, tile_size=100, plot_fig=True, train_ids=None, test_ids=None):
    try:
        opacity=0.3
        is_rgb=False
        with rasterio.open(feature_image_path) as f_src, rasterio.open(label_image_path) as l_src:
            features = f_src.read()  # (bands, H, W)
            labels = l_src.read() 
            if l_src.count != 1:
                is_rgb=True
                labels = l_src.read([4,1,2]) 
            labels = np.transpose(labels, (1, 2, 0))
        # if labels.shape[0] != 3:
        #     raise ValueError("Expected RGB label image with 3 bands.")

        if is_rgb:
            # Transpose to (H, W, 3) for visualization
            # label_rgb = np.transpose(labels, (1, 2, 0))

            p2 = np.percentile(labels, 2)
            p98 = np.percentile(labels, 98)

            label_rgb = np.clip(labels, p2, p98)  # Clip extremes
            label_rgb = (label_rgb - p2) / (p98 - p2) * 255
            labels = np.clip(label_rgb, 0, 255).astype(np.uint8)

        tiles = []
        positions = []

        for row in range(0, labels.shape[0], tile_size):
            for col in range(0, labels.shape[1], tile_size):
                if (row + tile_size) <= labels.shape[0] and (col + tile_size) <= labels.shape[1]:
                    feat_patch = features[:, row:row+tile_size, col:col+tile_size]
                    label_patch = labels[row:row+tile_size, col:col+tile_size, :]
                    tiles.append(((row, col), feat_patch, label_patch))
                    positions.append((row, col))

        if plot_fig:
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.imshow(labels)

            # if is_rgb:
            # else:
            #     ax.imshow(labels.squeeze())
            for idx, (row, col) in enumerate(positions):
                if idx in train_ids:
                    face_color = 'green'
                    opacity=0.4
                    
                elif idx in test_ids:
                    face_color = 'blue'
                    opacity=0.4
                    
                else:
                    # continue
                    face_color=None
                    opacity=0.1
                    
                                        
                rect = plt.Rectangle((col, row), tile_size, tile_size,
                                     linewidth=1, edgecolor="black",
                                     facecolor=face_color, alpha=opacity)
                ax.add_patch(rect)

                ax.text(col + tile_size // 2, row + tile_size // 2, str(idx),
                        fontsize=7, color='black', ha='center', va='center')
            print("113")

            legend_elements = [
                Patch(facecolor='green', edgecolor='green', label='Train'),
                Patch(facecolor='blue', edgecolor='blue', label='Test'),
                # Patch(facecolor='red', edgecolor='red', label='Other')
            ]
            ax.legend(handles=legend_elements, loc='upper right')
            plt.title("Train and Test Tile Selection")
            plt.tight_layout()
            plt.grid(False)
            ax.set_xticks([])  # Remove x-axis labels
            ax.set_yticks([])
            plt.show()

        print("Fishnet tile generation done")
        return tiles

    except Exception as e:
        print("Error in fishnet: ", str(e))


def tiles_to_samples(tile_list):
    try:
        X_list = []
        y_list = []
        for feat_patch, label_patch in tile_list:
            # Reshape each patch to (num_pixels, num_features)
            num_bands, h, w = feat_patch.shape
            X = feat_patch.reshape(num_bands, -1).T  # (bands, H*W) → (H*W, bands)
            y = label_patch.flatten()  # (H*W,)
            # print(f"Tile: feat_patch= {feat_patch.shape}, X={X.shape}, y={y.shape}")
            # Optional: remove pixels with label 0 (or nodata)
            # mask = y > 0  # Adjust this depending on your GT
            # X = X[mask]
            # y = y[mask]
            # print(X.shape,y.shape)
            X_list.append(X)
            y_list.append(y)
        
        return np.vstack(X_list), np.hstack(y_list)
    except:
        return [],[]


def prepare_training_sample(tiles, train_ids, test_ids):
    try:

        train_tiles = [tiles[i][1:] for i in train_ids]  # (feature_patch, label_patch)
        # val_tiles = [tiles[i][1:] for i in val_ids]
        test_tiles = [tiles[i][1:] for i in test_ids]

        # Prepare data
        X_train, y_train = tiles_to_samples(train_tiles)
        # X_val, y_val = tiles_to_samples(val_tiles)
        X_test, y_test = tiles_to_samples(test_tiles)
        return  X_train, y_train, X_test, y_test
    except Exception as e:
        print("error in preparing training sample: ", str(e))


# def run_model(feature_image_path,gt_image_path, sample_feature_path, feature_column_names, drop_columns, class_column_name ,models,output_model_dir, output_feat_imp_dir, corr_mat_dir, extended_file_name):
def run_model(feature_image_path,gt_image_path, feature_column_names,model_name, model,output_model_dir, output_feat_imp_dir, extended_file_name, train_ids, test_ids, tile_size=100):
    try:
        # metrics_combined=[]
        # for i in range(len(list(models))):
        # model_name=list(models.keys())[i]
        print(f"***************{model_name}*********************")
        # model=list(models.values())[i]
        tiles=create_fish_net(feature_image_path, gt_image_path, tile_size=tile_size,plot_fig=False)

        # train_ids = [1,3,5,7,9,15,17,20,21,23,27,29,30,32,33,34,36,40,41,42,43,44,45,46,47,48,53,57,58,59,60,61,62,67,68,70,71,73,74,75]
        # val_ids = []
        # test_ids = [4,10,19,22,31,35,55,56,69,72]

        # X_train, X_test, y_train, y_test = prepare_training_sample(sample_feature_path, feature_column_names, class_column_name,drop_columns, corr_mat_dir, model_name, extended_file_name)
        X_train, y_train, X_test, y_test = prepare_training_sample(tiles, train_ids,test_ids)
        print(X_train.shape,y_train.shape,"hbfhsdvchsdvch")

        if model_name=="XGB":
            model.fit(
                X_train, y_train,
                eval_set=[(X_test, y_test)],
                # early_stopping_rounds=30,
                # verbose=True
            )
        else:
            model.fit(X_train, y_train)

        #Make a prediction
        
        if train_ids:
            y_train_pred=model.predict(X_train)
            print('Model performance for Training set smooth')
            smoothed_pred = uniform_filter(y_train_pred.astype(float), size=5)
            # Convert back to 0/1
            y_train_pred_processed = (smoothed_pred > 0.5).astype(int)

            metrics_train=evaluate_model(y_train,y_train_pred_processed)

        if test_ids:
            y_test_pred=model.predict(X_test)
            # print('Model performance for Test set')
            # metrics_test=evaluate_model(y_test,y_test_pred)

            print("--------------")
            print('Model performance for Test set smooth')
            smoothed_pred = uniform_filter(y_test_pred.astype(float), size=5)
            # Convert back to 0/1
            y_test_pred_processed = (smoothed_pred > 0.5).astype(int)
            metrics_test=evaluate_model(y_test,y_test_pred_processed)
            print("----------------------------------------")
        # if val_ids:
        #     y_val_pred=model.predict(X_val)
        #     print('Model performance for Val set')
        #     metrics_test=evaluate_model(y_val,y_val_pred)
        #     print("----------------------------------------")


        metrics={
        'acc_train': round(metrics_train['accuracy'],4),
        'f1_train': round(metrics_train['f1_score'],4),
        'precision_train': round(metrics_train['precision'],4),
        'recall_train': round(metrics_train['recall'],4),
        'roc_auc_train': round(metrics_train['roc_auc'],4),
        'acc_test': round(metrics_test['accuracy'],4),
        'f1_test': round(metrics_test['f1_score'],4),
        'precision_test': round(metrics_test['precision'],4),
        'recall_test': round(metrics_test['recall'],4),
        'roc_auc_test': round(metrics_test['roc_auc'],4),
        }
        # metrics_combined.append(metrics)

        # Assuming you have a trained model called `model`
        joblib.dump(model, f'{output_model_dir}{model_name}{extended_file_name}.pkl')
        print("Model saved successfully!")

        # # Plot a simple bar chart
        feature_importances = pd.Series(model.feature_importances_, index=feature_column_names).sort_values(ascending=False)
        # feature_importances = pd.Series(model.feature_importances_, index=X_train.columns).sort_values(ascending=False)
        feature_importances.plot.bar()
        plt.savefig(f"{output_feat_imp_dir}{model_name}{extended_file_name}.png", dpi=300, bbox_inches="tight")  # Change filename & format as needed
        return metrics

        # predict(model, feature_image_path, model_name)
    except Exception as e:
        print(f"error in run model: ", str(e))


def predict(model, image_path, output_file_path,output_image_path,title ):
    dataset = rasterio.open(image_path)
    print(dataset.count)

    # Read the image bands into an array (assuming it's a multi-band raster)
    sar_bands = np.stack([dataset.read(i+1) for i in range(dataset.count)], axis=-1)

    # Reshape the SAR data to a 2D array (num_pixels, num_bands)
    height, width, num_bands = sar_bands.shape
    pixels = sar_bands.reshape(-1, num_bands)  # Each row is a pixel

    # Preprocess (scale) the pixel values if needed (based on training data preprocessing)
    # scaler = StandardScaler()
    # pixels_scaled = scaler.fit_transform(pixels)  # Apply scaling to the pixels

    # Predict for each pixel
    predictions = model.predict(pixels)
    print(np.unique(predictions))

    # Assuming predictions are a 2D array (for an image or spatial data)
    predictions = uniform_filter(predictions, size=4)  # size is the window size

    # predictions = model.predict(scaler.transform(pixels))
    # Binary classification: make sure values are 0 and 1
    # binary_prediction = (predictions > 0.5).astype(int)  # You may adjust threshold

    # # 🧹 Remove small objects (<100 pixels)
    # cleaned_prediction = remove_small_objects(binary_prediction.astype(bool), min_size=10)

    # # Convert back to integer image
    # predictions = cleaned_prediction.astype(np.uint8)


    # Reshape predictions to match the image dimensions
    predicted_image = predictions.reshape(height, width)

    # Save the predicted classes to a new file
    # output_path = f"output/prediction/{prefix}_{model_name}_{extended_file_name}.tif"
    meta = dataset.meta
    meta.update(dtype=rasterio.uint8, count=1)  # Assuming class labels are integers, uint8 works for this

    with rasterio.open(output_file_path, 'w', **meta) as dest:
        dest.write(predicted_image.astype(rasterio.uint8), 1)  # Write to the first band

    print(f"Predictions saved to {output_file_path}")


    pred_dataset = rasterio.open(output_file_path)
    # Read the first band of the image (you can adjust for multi-band images)
    pred_band_1 = pred_dataset.read(1)
    row_start, row_end = 100, 1400
    col_start, col_end = 550, 2550
    # Crop the classified image
    clipped_img = pred_band_1[row_start:row_end, col_start:col_end]
    # Plot the image
    plt.figure(figsize=(6, 6))
    plt.axis('off')
    cmap = mcolors.ListedColormap(['#1dc47c', '#c41d28'])

    # Plot the image with the custom colormap
    plt.imshow(clipped_img, cmap=cmap, interpolation='nearest')
    plt.title(title)
    # Create custom legend
    legend_patches = [
        mpatches.Patch(color='#c41d28', label='Burnt'),
        mpatches.Patch(color='#1dc47c', label='Non-Burnt')
    ]

    # plt.legend(handles=legend_patches, loc='lower right', frameon=True)
    plt.savefig(output_image_path,bbox_inches='tight', dpi=300)

    plt.show()


def glcm_single(input_path, output_path,start_index,end_index):

    with rasterio.open(input_path) as src:
        meta = src.meta.copy()
        meta.update(count=20)  # output will have 20 bands

        # Read all 60 bands
        bands = src.read()  # shape: (60, height, width)

        # Prepare the output stack
        # averaged_stack = np.zeros((20, src.height, src.width), dtype=full_stack.dtype)

        # for i in range(20):
        #     band1 = full_stack[i, :, :]
        #     band2 = full_stack[i + 20, :, :]
        #     band3 = full_stack[i + 40, :, :]

        #     averaged_stack[i] = (band1 + band2 + band3) / 3

    # Write the output raster
    index=1
    with rasterio.open(output_path, 'w', **meta) as dst:
        for i in range(start_index,end_index):
            dst.write(bands[i],index)
            index=index+1

    print(f"Averaged raster saved as: {output_path}")



def glcm_average(input_path, output_path):

    with rasterio.open(input_path) as src:
        meta = src.meta.copy()
        meta.update(count=20)  # output will have 20 bands

        # Read all 60 bands
        full_stack = src.read()  # shape: (60, height, width)

        # Prepare the output stack
        averaged_stack = np.zeros((20, src.height, src.width), dtype=full_stack.dtype)

        for i in range(20):
            band1 = full_stack[i, :, :]
            band2 = full_stack[i + 20, :, :]
            band3 = full_stack[i + 40, :, :]

            averaged_stack[i] = (band1 + band2 + band3) / 3

    # Write the output raster
    with rasterio.open(output_path, 'w', **meta) as dst:
        dst.write(averaged_stack)

    print(f"Averaged raster saved as: {output_path}")


def compute_pca(input_image, output_path,n_components=3):

    with rasterio.open(input_image) as src:
        bands = src.read()  # Shape will be (bands, height, width)
        meta=src.meta.copy()

    pixels = bands.reshape(bands.shape[0], -1).T  # Transpose to have shape (num_pixels, num_bands)
    pixels = np.nan_to_num(pixels, nan=0.0) 
    print(np.min(pixels),np.max(pixels))

    pca = PCA(n_components=n_components)  # Get the first 5 principal components
    principal_components = pca.fit_transform(pixels)  # Shape will be (num_pixels, 5)

    num_components = pca.n_components_
    print(f'Number of principal components selected: {num_components}')

    principal_components_image = principal_components.T.reshape(num_components, bands.shape[1], bands.shape[2])

    print(np.min(principal_components_image[0]),np.max(principal_components_image[0]))
    plt.figure(figsize=(15, 15))
    for i in range(num_components):
        plt.subplot(1, num_components, i+1)
        plt.imshow(principal_components_image[i], cmap='gray')
        plt.title(f'Principal Component {i+1}')
    plt.show()

    # Step 6: Save the PCA results to a new TIFF file
    meta.update({'count':num_components})

    with rasterio.open(output_path, 'w', **meta) as dst:
        # Write each principal component as a separate band
        for i in range(num_components):
            dst.write(principal_components_image[i], i+1)

def compute_dglcm(pre_image_path, post_image_path,output_path):
    with rasterio.open(pre_image_path) as pre_src, rasterio.open(post_image_path) as post_src:
        pre = pre_src.read()   # shape: (bands, height, width)
        post = post_src.read()
        diff = post - pre

        meta = pre_src.meta.copy()
        meta.update(dtype=rasterio.float32)

    with rasterio.open(output_path, 'w', **meta) as dst:
        dst.write(diff.astype(np.float32))

    print(f"Saved dglcm to {output_path}")

def compute_asc_desc_dglcm(asc_image_path, desc_image_path,output_path):
    with rasterio.open(asc_image_path) as pre_src, rasterio.open(desc_image_path) as post_src:
        pre = pre_src.read()   # shape: (bands, height, width)
        post = post_src.read()
        # combined = np.maximum(pre,post)
        combined = (pre+post)/2


        meta = pre_src.meta.copy()
        meta.update(dtype=rasterio.float32)

    with rasterio.open(output_path, 'w', **meta) as dst:
        dst.write(combined.astype(np.float32))

    print(f"Saved dglcm to {output_path}")


def get_bbox(aoi_path):
    aoi_gdf = gpd.read_file(aoi_path)

    # Extract the bounding box (minx, miny, maxx, maxy)
    bbox = aoi_gdf.total_bounds  # This gives the bounding box as a list [minx, miny, maxx, maxy]

    print("Bounding Box:", bbox)
    return bbox


def get_best_hyperparameter(random_search_model,feature_image_path,gt_image_path,rf_params,train_ids,tile_size):
    # Define parameter grid
    # rf_params = {
    #     'n_estimators': [50, 100, 150],  # Number of trees
    #     'max_depth': [10, 20,30],  # Depth of trees
    #     'min_samples_split': [10, 20, 40],  # Minimum samples to split
    #     'max_features': [2, 4, 'sqrt', 'log2'],  # Features to consider at each split
    #     'class_weight': ['balanced_subsample', 'balanced'],  # Handle class imbalance
    #     'max_samples': [0.3,0.5, 0.7],  # Fraction of samples to train each tree on
    # }

    # Initialize GridSearchCV with parameter grid
    # grid_search = RandomizedSearchCV(estimator=RandomForestClassifier(), param_grid=rf_params, cv=3, verbose=2, n_jobs=-1)
    # grid_search = RandomizedSearchCV(estimator=RandomForestClassifier(), param_distributions=rf_params, cv=3, verbose=2, n_jobs=-1)

    # feature_image_path="MachineLearning/output/feature_image/palisades_sar_avg_asc_desc.tif"
    # gt_image_path="MachineLearning/gt/palisades_label_0_17.tif"
    cv_split = KFold(n_splits=5, random_state=42, shuffle=True)

    rf_random_search=RandomizedSearchCV(
        estimator=random_search_model,
        param_distributions=rf_params,
        n_iter=200,
        scoring="neg_log_loss",
        refit=True,
        return_train_score=True,
        cv=cv_split,    
        verbose=10,
        n_jobs=-1,
        random_state=42
    )
    tiles=create_fish_net(feature_image_path, gt_image_path, tile_size=tile_size,plot_fig=False)

    # train_ids = [154, 351,345,340, 184, 478,355,178,368,88,172,439,303,219,375,435]
    # train_ids = [146,129,123,344,378,248,132,189,242,297,321,235,356]
    test_ids=[]

    # X_train, X_test, y_train, y_test = prepare_training_sample(sample_feature_path, feature_column_names, class_column_name,drop_columns, corr_mat_dir, model_name, extended_file_name)
    X_train, y_train, X_test, y_test = prepare_training_sample(tiles, train_ids,test_ids)
    # print(X_train.shape,y_train.shape,"hbfhsdvchsdvch")

    # Fit the model
    rf_random_search.fit(X_train, y_train)

    # Print the best parameters found
    print(rf_random_search.best_params_)
