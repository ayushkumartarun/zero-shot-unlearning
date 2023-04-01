from sklearn.svm import SVC

def entropy(p, dim = -1, keepdim = False):
    return -torch.where(p > 0, p * p.log(), p.new([0.0])).sum(dim=dim, keepdim=keepdim)

def collect_prob(data_loader, model):   
    data_loader = torch.utils.data.DataLoader(data_loader.dataset, batch_size=1, shuffle=False, num_workers = 32, prefetch_factor = 10)
    prob = []
    with torch.no_grad():
        for batch in data_loader:
            batch = [tensor.to(next(model.parameters()).device) for tensor in batch]
            data, _, target = batch
            output = model(data)
            prob.append(F.softmax(output, dim=-1).data)
    return torch.cat(prob)

def get_membership_attack_data(retain_loader, forget_loader, test_loader, model):    
    retain_prob = collect_prob(retain_loader, model)
    forget_prob = collect_prob(forget_loader, model)
    test_prob = collect_prob(test_loader, model)
    
    X_r = torch.cat([entropy(retain_prob), entropy(test_prob)]).cpu().numpy().reshape(-1, 1)
    Y_r = np.concatenate([np.ones(len(retain_prob)), np.zeros(len(test_prob))])
    
    X_f = entropy(forget_prob).cpu().numpy().reshape(-1, 1)
    Y_f = np.concatenate([np.ones(len(forget_prob))])    
    return X_f, Y_f, X_r, Y_r

def get_membership_attack_prob(retain_loader, forget_loader, test_loader, model):
    X_f, Y_f, X_r, Y_r = get_membership_attack_data(retain_loader, forget_loader, test_loader, model)
    clf = SVC(C=3,gamma='auto',kernel='rbf')
    #clf = LogisticRegression(class_weight='balanced',solver='lbfgs',multi_class='multinomial')
    clf.fit(X_r, Y_r)
    results = clf.predict(X_f)
    return results.mean()

def relearn_time(model, train_loader, valid_loader, reqAcc, lr):
    # measuring relearn time for gold standard model
    rltime = 0
    curr_Acc = 0
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    
    # we will try the relearning step till 4 epochs.
    for epoch in range(10):
        
        for batch in train_loader:
            model.train()
            loss = training_step(model, batch)
            loss.backward()
            
            optimizer.step()
            optimizer.zero_grad()
            history = [evaluate(model, valid_dl)]
            curr_Acc = history[0]["Acc"]*100
            print(curr_Acc, sep=',')
            
            
            
            rltime += 1
            if(curr_Acc >= reqAcc):
                break
                
        if(curr_Acc >= reqAcc):
            break
    return rltime

def ain(full_model, model, gold_model, train_data, val_retain, val_forget, 
                  batch_size = 256, error_range = 0.05, lr = 0.001):
    # measuring performance of fully trained model on forget class
    forget_valid_dl = DataLoader(val_forget, batch_size)
    history = [evaluate(full_model, forget_valid_dl)]
    AccForget = history[0]["Acc"]*100
    
    print("Accuracy of fully trained model on forget set is: {}".format(AccForget))
    
    retain_valid_dl = DataLoader(val_retain, batch_size)
    history = [evaluate(full_model, retain_valid_dl)]
    AccRetain = history[0]["Acc"]*100
    
    print("Accuracy of fully trained model on retain set is: {}".format(AccRetain))
    
    history = [evaluate(model, forget_valid_dl)]
    AccForget_Fmodel = history[0]["Acc"]*100
    
    print("Accuracy of forget model on forget set is: {}".format(AccForget_Fmodel))
    
    history = [evaluate(model, retain_valid_dl)]
    AccRetain_Fmodel = history[0]["Acc"]*100
    
    print("Accuracy of forget model on retain set is: {}".format(AccRetain_Fmodel))
    
    history = [evaluate(gold_model, forget_valid_dl)]
    AccForget_Gmodel = history[0]["Acc"]*100
    
    print("Accuracy of gold model on forget set is: {}".format(AccForget_Gmodel))
    
    history = [evaluate(gold_model, retain_valid_dl)]
    AccRetain_Gmodel = history[0]["Acc"]*100
    
    print("Accuracy of gold model on retain set is: {}".format(AccRetain_Gmodel))
    
    reqAccF = (1-error_range)*AccForget
    
    print("Desired Accuracy for retrain time with error range {} is {}".format(error_range, reqAccF))
    
    train_loader = DataLoader(train_ds, batch_size, shuffle = True)
    valid_loader = DataLoader(val_forget, batch_size)
    rltime_gold = relearn_time(model = gold_model, train_loader = train_loader, valid_loader = valid_loader, 
                               reqAcc = reqAccF,  lr = lr)
    
    print("Relearning time for Gold Standard Model is {}".format(rltime_gold))
    
    rltime_forget = relearn_time(model = model, train_loader = train_loader, valid_loader = valid_loader, 
                               reqAcc = reqAccF, lr = lr)
    
    print("Relearning time for Forget Model is {}".format(rltime_forget))
    
    rl_coeff = rltime_forget/rltime_gold
    print("AIN = {}".format(rl_coeff))