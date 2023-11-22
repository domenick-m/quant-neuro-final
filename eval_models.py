# from joblib import load; ridge_model = load('ridge_model.joblib')
# self.load_state_dict(torch.load(filepath))

#%%
# # create test set
# test_dict = make_eval_input_tensors(dataset, dataset_name=dataset_name, trial_split='val', save_file=False)
# test_spikes = np.concatenate((test_dict['eval_spikes_heldin'], test_dict['eval_spikes_heldout']), axis=2)
# test_behavior = behav_dict['mc_maze']['eval_behavior']

# print(test_spikes.shape)
# print(test_behavior.shape)


#%% ----------------------------------------------------------------------------
# Evaluate the trained model
# ------------------------------------------------------------------------------
# outputs, gt_behav = [], []
# from torcheval.metrics.functional import r2_score

# model.eval()
# for spikes, behavior in zip(val_spikes_tensor, val_behavior_tensor):
#     val_loss, val_r2 = 0, 0
#     with torch.no_grad():
#         spikes, behavior = spikes.to(device), behavior.to(device)
#         output = model(spikes)
#         outputs.append(output)
#         gt_behav.append(behavior)

#         loss = mse_loss(output, behavior)
#         val_loss += loss.item()

#         r2_val = r2_score(output, behavior, multioutput='uniform_average')
#         val_r2 += r2_val.item()


#%% ----------------------------------------------------------------------------
# # Plot some trials
# # ------------------------------------------------------------------------------
# trials = [0, 1, 2, 3, 4]
# fig, axs = plt.subplots(2, 5, figsize=(40, 10))
# for idx, i in enumerate(range(5)):
#     axs[0, idx].plot(outputs[i].cpu().numpy()[:, 0], label='pred')
#     axs[0, idx].plot(gt_behav[i].cpu().numpy()[:, 0], label='gt')
#     axs[0, idx].legend()
#     axs[0, idx].set_title(f'X Velocity - Trial:{i}')
#     axs[1, idx].plot(outputs[i].cpu().numpy()[:, 1], label='pred')
#     axs[1, idx].plot(gt_behav[i].cpu().numpy()[:, 1], label='gt')
#     axs[1, idx].set_title(f'Y Velocity - Trial:{i}')
# plt.show()