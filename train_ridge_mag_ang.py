 # split into training and validation sets
    tv_splits = train_test_split(trainval_smth_spikes, 
                                trainval_behavior, 
                                test_size=0.2, 
                                random_state=1)
    train_spikes, val_spikes, train_behavior, val_behavior = tv_splits

    return (
        (train_spikes, train_behavior), 
        (val_spikes, val_behavior), 
        (test_smth_spikes, test_behavior)
    )