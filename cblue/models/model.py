import torch
import torch.nn as nn


class ERModel(nn.Module):
    def __init__(self, encoder_class, encoder_path):
        super(ERModel, self).__init__()
        self.encoder = encoder_class.from_pretrained(encoder_path)
        self.sub_startlayer = nn.Linear(in_features=self.encoder.config.hidden_size, out_features=1)
        self.sub_endlayer = nn.Linear(in_features=self.encoder.config.hidden_size, out_features=1)
        self.obj_startlayer = nn.Linear(in_features=self.encoder.config.hidden_size, out_features=1)
        self.obj_endlayer = nn.Linear(in_features=self.encoder.config.hidden_size, out_features=1)

    def forward(self, input_ids, token_type_ids, attention_mask):
        output = self.encoder(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        last_hidden_state = output[0]   # batch, seq, hidden

        sub_start_idx = self.sub_startlayer(last_hidden_state).sigmoid()
        sub_end_idx = self.sub_endlayer(last_hidden_state).sigmoid()
        obj_start_idx = self.obj_startlayer(last_hidden_state).sigmoid()
        obj_end_idx = self.obj_endlayer(last_hidden_state).sigmoid()

        return sub_start_idx.squeeze(-1), sub_end_idx.squeeze(-1), \
               obj_start_idx.squeeze(-1), obj_end_idx.squeeze(-1)


class REModel(nn.Module):
    def __init__(self, tokenizer, encoder_class, encoder_path, num_labels):
        super(REModel, self).__init__()
        self.bert = encoder_class.from_pretrained(encoder_path)
        self.bert.resize_token_embeddings(len(tokenizer))
        self.classifier = nn.Linear(in_features=self.bert.config.hidden_size*2, out_features=num_labels)

    def forward(self, input_ids, token_type_ids, attention_mask, flag, labels=None):
        device = input_ids.device

        output = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        last_hidden_state = output[0]   # batch, seq, hidden
        batch_size, seq_len, hidden_size = last_hidden_state.shape
        entity_hidden_state = torch.Tensor(batch_size, 2*hidden_size) # batch, 2*hidden
        # flag: batch, 2
        for i in range(batch_size):
            sub_start_idx, obj_start_idx = flag[i, 0], flag[i, 1]
            start_entity = last_hidden_state[i, sub_start_idx, :].view(hidden_size, )   # s_start: hidden,
            end_entity = last_hidden_state[i, obj_start_idx, :].view(hidden_size, )   # o_start: hidden,
            entity_hidden_state[i] = torch.cat([start_entity, end_entity], dim=-1)
        entity_hidden_state = entity_hidden_state.to(device)
        logits = self.classifier(entity_hidden_state)
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            return loss_fn(logits, labels.view(-1)), logits
        return logits


class CLSModel(nn.Module):
    def __init__(self, encoder_class, encoder_path, num_labels):
        super(CLSModel, self).__init__()

        self.encoder = encoder_class.from_pretrained(encoder_path)
        self.num_labels = num_labels
        self.dropout = nn.Dropout(self.bert.config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                labels=None,
                output_hidden_states=None):
        outputs = self.encoder(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask,
                               output_hidden_states=output_hidden_states)
        # batch, seq, hidden
        last_hidden_states, first_hidden_states = outputs[0], outputs[2][0]
        # batch, hidden
        avg_hidden_states = torch.mean((last_hidden_states + first_hidden_states), dim=1)
        avg_hidden_states = self.dropout(avg_hidden_states)
        logits = self.classifier(avg_hidden_states)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss, logits

        return logits
