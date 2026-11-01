import os
import glob
import pandas as pd

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import albumentations as A


MEAN = [0.485, 0.456, 0.406]
STD=[0.229, 0.224, 0.225]

# Required constants.
#TRAIN_DIR = os.path.join('input/Cotton-Disease-Training/trainning/Cotton leaves - Training/800 Images')
#VALID_DIR = os.path.join('input/Cotton-Disease-Validation/validation/Cotton plant disease-Validation/Cotton plant disease-Validation')

# Training transforms
def get_train_transform(input_shape):
    
    #train_transform = transforms.Compose([
    #    transforms.Resize((image_size, image_size)),
    #    transforms.RandomHorizontalFlip(p=0.5),
    #    transforms.RandomRotation(35),
    #    transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
    #    transforms.ToTensor(),
    #    transforms.Normalize(
    #        mean=[0.485, 0.456, 0.406],
    #        std=[0.229, 0.224, 0.225]
    #        )
    #])

    train_transform = A.Compose([
        A.Resize(input_shape[1], input_shape[0], always_apply=True),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=35),
        A.RandomBrightnessContrast(p=0.5),
        A.Normalize(mean=MEAN, std=STD)
    ], is_check_shapes=False)

    return train_transform

# Validation transforms
def get_valid_transform(input_shape):
    #valid_transform = transforms.Compose([
    #    transforms.Resize((image_size, image_size)),
    #    transforms.ToTensor(),
    #    transforms.Normalize(
    #        mean=[0.485, 0.456, 0.406],
    #        std=[0.229, 0.224, 0.225]
    #        )
    #])

    valid_transform = A.Compose([
        A.Resize(input_shape[1], input_shape[0], always_apply=True),
        A.Normalize(mean=MEAN, std=STD)
    ], is_check_shapes=False)

    return valid_transform

# Orig transforms
def get_orig_transform(input_shape):
    #valid_transform = transforms.Compose([
    #    transforms.Resize((image_size, image_size)),
    #    transforms.ToTensor(),
    #    transforms.Normalize(
    #        mean=[0.485, 0.456, 0.406],
    #        std=[0.229, 0.224, 0.225]
    #        )
    #])

    orig_transform = A.Compose([
        A.Resize(input_shape[1], input_shape[0], always_apply=True),
        #A.Normalize(mean=MEAN, std=STD)
    ], is_check_shapes=False)

    return orig_transform
    
#def get_datasets():
#    """
#    Function to prepare the Datasets.
#    Returns the training and validation datasets along 
#    with the class names.
#    """
#    dataset_train = datasets.ImageFolder(
#        TRAIN_DIR, 
#        transform=(get_train_transform(IMAGE_SIZE))
#    )
#    dataset_valid = datasets.ImageFolder(
#        VALID_DIR, 
#        transform=(get_valid_transform(IMAGE_SIZE))
#    )
#    return dataset_train, dataset_valid, dataset_train.classes

def get_datasets(path_dataset_directory, 
    dataset_name="dataset_224_images_in_train_class", 
	image_dir_for_training="images",
	classes_to_avoid_during_training=[], 
	dataset_images_list="dataset_images_list"):
	directory_dataset = os.path.join(path_dataset_directory, dataset_images_list, dataset_name)
	directory_images = os.path.join(path_dataset_directory, image_dir_for_training)
	all_images_lists = glob.glob(directory_dataset+os.sep+"*", recursive = False)

	file_images_list_train = next((file for file in all_images_lists if file.endswith("_train.csv")), None)
	file_images_list_valid = next((file for file in all_images_lists if file.endswith("_valid.csv")), None)
	#file_images_list_test = next((file for file in all_images_lists if file.endswith("_test.csv")), None)

	if file_images_list_train == None:
		assert False, ('File containing list of training images not found')
	if file_images_list_valid == None:
		assert False, ('File containing list of validation images not found')
	#if file_images_list_test == None:
	#	assert False, ('File containing list of test images not found')

	df_train = pd.read_csv(file_images_list_train, sep=";")  
	df_valid = pd.read_csv(file_images_list_valid, sep=";")  
	#df_test = pd.read_csv(os.path.join(directory_dataset, file_images_list_test), sep=";")  
	
	key_class = df_train.keys()[-2]#"State" or "Stereotype"
	
	# Remove classes to not use in the image list 
	if classes_to_avoid_during_training:
		for class_to_remove in classes_to_avoid_during_training:
			df_train = df_train[df_train[key_class]!=class_to_remove]
			df_valid = df_valid[df_valid[key_class]!=class_to_remove]
			
	dataset_classes = list(df_train[key_class].unique())
	train_images = df_train['image_path'].tolist()
	train_images = [os.path.join(directory_images, image_path) for image_path in train_images]
	train_labels = [dataset_classes.index(value) for value in df_train[key_class].tolist()]

	valid_images = df_valid['image_path'].tolist()
	valid_images = [os.path.join(directory_images, image_path) for image_path in valid_images]	
	valid_labels = [dataset_classes.index(value) for value in df_valid[key_class].tolist()]

	if 'processed_images' in image_dir_for_training:
		train_images = [item.replace('Wildtype-not-on-border', 'Wildtype') if 'Wildtype-not-on-border' in item else item for item in train_images]
		train_images = [item.replace('Wildtype-on-border', 'Wildtype') if 'Wildtype-on-border' in item else item for item in train_images]
		valid_images = [item.replace('Wildtype-not-on-border', 'Wildtype') if 'Wildtype-not-on-border' in item else item for item in valid_images]
		valid_images = [item.replace('Wildtype-on-border', 'Wildtype') if 'Wildtype-on-border' in item else item for item in valid_images]


	return train_images, train_labels, valid_images, valid_labels, dataset_classes


def get_dict_of_forecasted_classes_to_change(expected_classes, got_classes):
    classes_to_change = {}
    for i in range(len(got_classes)):
        # Check of differences
        if got_classes[i] != expected_classes[i]: 
            index_to_change = expected_classes.index(got_classes[i])
            # Change forecasted value "i" by the value "index_to_change"
            classes_to_change[i] = index_to_change
    return classes_to_change

def get_dataset_test(path_dataset_directory, dataset_name="dataset_224_images_in_train_class", 
	image_dir_for_training="images",
	expected_classes=['State_1', 'State_2', 'State_3', 'State_4', 'State_5', 'State_6'],
	dataset_images_list = "dataset_images_list"):
	
    directory_dataset = os.path.join(path_dataset_directory, dataset_images_list, dataset_name)
    directory_images = os.path.join(path_dataset_directory, image_dir_for_training)
    all_images_lists = glob.glob(directory_dataset+os.sep+"*", recursive = False)

    file_images_list_test = next((file for file in all_images_lists if file.endswith("_test.csv")), None)

    if file_images_list_test == None:
       print("directory_dataset: "+directory_dataset)
       print("directory_images: "+directory_images)
       print("all_images_lists: "+str(all_images_lists))
       assert False, ('File containing list of test images not found')

    df_test = pd.read_csv(file_images_list_test, sep=";")  
    
    key_class = df_test.keys()[-2]#"State"

    dataset_classes = list(df_test[key_class].unique())

    test_images = df_test['image_path'].tolist()
    test_images = [os.path.join(directory_images, image_path) for image_path in test_images]  
    test_labels = [dataset_classes.index(value) for value in df_test[key_class].tolist()]

    classes_to_change = get_dict_of_forecasted_classes_to_change(expected_classes, dataset_classes)
    # Change labels according to the expected one

    if classes_to_change:
    	test_labels = [classes_to_change.get(item, item) for item in test_labels]

    if 'processed_images' in image_dir_for_training:
    	test_images = [item.replace('Wildtype-not-on-border', 'Wildtype') if 'Wildtype-not-on-border' in item else item for item in test_images]
    	test_images = [item.replace('Wildtype-on-border', 'Wildtype') if 'Wildtype-on-border' in item else item for item in test_images]
    	
    return test_images, test_labels, dataset_classes


def get_data_loaders(dataset_train, dataset_valid, batch_size=16, pin_memory=False,
    train_sampler = None, val_sampler = None, shuffle_train=True,
    drop_last_train = True, drop_last_valid = False, num_workers=4):
    """
    Prepares the training and validation data loaders.
    :param dataset_train: The training dataset.
    :param dataset_valid: The validation dataset.
    Returns the training and validation data loaders.
    """
    train_loader = DataLoader(
        dataset_train, batch_size=batch_size, 
        shuffle=shuffle_train, num_workers=num_workers,
        drop_last=drop_last_train, 

        # Pin memory for the GPU
        pin_memory=pin_memory,

        # Sampler for Distributed Dataloader
        sampler=train_sampler
    )
    valid_loader = DataLoader(
        dataset_valid, batch_size=batch_size, 
        shuffle=False, num_workers=num_workers,
        drop_last=drop_last_valid, 

        # Pin memory for the GPU 
        pin_memory=pin_memory,

        # Sampler for Distributed Dataloader
        sampler=val_sampler
    )
    
    return train_loader, valid_loader 

def get_data_loader_valid(dataset_valid, batch_size, num_workers=4):
    """
    Prepares the validation data loaders.
    :param dataset_valid: The validation dataset.
    Returns the training and validation data loaders.
    """
    valid_loader = DataLoader(
        dataset_valid, batch_size=batch_size, 
        shuffle=False, num_workers=num_workers
    )
    
    return valid_loader 


def get_data_loader_test(dataset_test, batch_size, num_workers=4):
    """
    Prepares the training and validation data loaders.
    :param dataset_train: The training dataset.
    :param dataset_valid: The validation dataset.
    Returns the training and validation data loaders.
    """
    test_loader = DataLoader(
        dataset_test, batch_size=batch_size, 
        shuffle=False, num_workers=num_workers
    )
    
    return test_loader


################################################################
def load_model_name_and_class_to_change(device, pth_model_file_path, pth_parameters_file_path, CLASSES_NAMES):
    #Load performances and parameters of the model locally on the CPU
    data_model = torch.load(pth_model_file_path, map_location=torch.device('cpu'))

    #if 'pretrained' not in data_model.keys() or "any_res" in data_parameters['pretrained_model_name_or_path']:
    #    data_model['pretrained'] = True

    
    # Open the file in binary read mode
    img_black_and_white = False
    apply_mask = False
    with open(pth_parameters_file_path, 'rb') as file:
        # Load the object from the file
        data_parameters = pickle.load(file)
                
        # Indicates if black and white images are used
        if data_parameters['img_black_and_white']:
            img_black_and_white = True

        # Indicates if a mask is applied for remove background information
        if data_parameters['apply_mask']:
            apply_mask = True
    
    # Load the model.
    model = build_model(
            num_classes = len(data_model['dataset_classes']), 
            input_shape = data_model['input_shape'],
            pretrained_model_name_or_path = data_model['pretrained_model_name_or_path'],
            in_features = data_model['in_features'],
            pretrained = False #data_model['pretrained']

    )#.to(device)
        
    # Put model on the gpu
    model = model.to(device)
    
    # load weights
    model.load_state_dict(data_model['state_dict'])
    model.eval()
    
    #### Check if they are classes to change according to the expected values
    classes_to_change = get_dict_of_forecasted_classes_to_change(CLASSES_NAMES, data_model['dataset_classes'])

    # Get the dataset used for training
    image_dir_for_training = data_model['image_dir_for_training']

    # Get image input shape used for training
    input_shape = data_model['input_shape']
    
    return model, classes_to_change, image_dir_for_training, img_black_and_white, apply_mask, input_shape
    