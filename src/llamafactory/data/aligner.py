# Copyright 2024 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from functools import partial
from typing import TYPE_CHECKING, Any, Dict, List, Union

from datasets import Features
from sympy import im
from datasets import Value

from ..extras.logging import get_logger
from .data_utils import Role


if TYPE_CHECKING:
    from datasets import Dataset, IterableDataset
    from transformers import Seq2SeqTrainingArguments

    from ..hparams import DataArguments
    from .parser import DatasetAttr


logger = get_logger(__name__)


def _convert_images(images: List[Any], dataset_attr: "DatasetAttr", data_args: "DataArguments") -> List[Any]:
    r"""
    Optionally concatenates image path to dataset dir when loading from local disk.
    """
    outputs = []
    if dataset_attr.load_from in ["script", "file"]:
        for image in images:
            if isinstance(image, str) and os.path.isfile(os.path.join(data_args.dataset_dir, image)):
                outputs.append(os.path.join(data_args.dataset_dir, image))
            else:
                outputs.append(image)

    return outputs


def convert_alpaca(
    examples: Dict[str, List[Any]], dataset_attr: "DatasetAttr", data_args: "DataArguments"
) -> Dict[str, List[Any]]:
    r"""
    Converts alpaca format dataset to the standard format.
    """
    outputs = {"prompt": [], "response": [], "system": [], "tools": [], "images": []}

    # TODO:
    if data_args.channel_loss:
        outputs = {"prompt": [], "response": [], "system": [], "tools": [], "images": [], "channels": []}

    convert_images = partial(_convert_images, dataset_attr=dataset_attr, data_args=data_args)

    # print(dataset_attr.prompt) # instruction

    for i in range(len(examples[dataset_attr.prompt])):
        prompt = []
        if dataset_attr.history and isinstance(examples[dataset_attr.history][i], list):
            for old_prompt, old_response in examples[dataset_attr.history][i]:
                prompt.append({"role": Role.USER.value, "content": old_prompt})
                prompt.append({"role": Role.ASSISTANT.value, "content": old_response})

        content = []
        if dataset_attr.prompt and examples[dataset_attr.prompt][i]:
            content.append(examples[dataset_attr.prompt][i])

        if dataset_attr.query and examples[dataset_attr.query][i]:
            content.append(examples[dataset_attr.query][i])

        prompt.append({"role": Role.USER.value, "content": "\n".join(content)})  # "prompt\nquery"

        if dataset_attr.kto_tag and isinstance(examples[dataset_attr.kto_tag][i], bool):  # kto example
            response = [{"role": Role.ASSISTANT.value, "content": examples[dataset_attr.response][i]}]
            if examples[dataset_attr.kto_tag][i]:
                response = response + [{"role": Role.ASSISTANT.value, "content": ""}]
            else:
                response = [{"role": Role.ASSISTANT.value, "content": ""}] + response
        elif (
            dataset_attr.ranking
            and isinstance(examples[dataset_attr.chosen][i], str)
            and isinstance(examples[dataset_attr.rejected][i], str)
        ):  # pairwise example
            response = [
                {"role": Role.ASSISTANT.value, "content": examples[dataset_attr.chosen][i]},
                {"role": Role.ASSISTANT.value, "content": examples[dataset_attr.rejected][i]},
            ]
        elif dataset_attr.response and isinstance(examples[dataset_attr.response][i], str):  # normal example
            response = [{"role": Role.ASSISTANT.value, "content": examples[dataset_attr.response][i]}]
        else:  # unsupervised
            response = []

        outputs["prompt"].append(prompt)
        outputs["response"].append(response)
        outputs["system"].append(examples[dataset_attr.system][i] if dataset_attr.system else "")
        outputs["tools"].append(examples[dataset_attr.tools][i] if dataset_attr.tools else "")
        outputs["images"].append(convert_images(examples[dataset_attr.images][i]) if dataset_attr.images else [])

        # TODO: add channel
        if data_args.channel_loss:
            outputs["channels"].append([examples["channel"][i]])


    """
    # from threading import Lock
    # lock = Lock()
    # with lock:
    #     print("prompt: ", outputs["prompt"][0])
    #     print("response: ", outputs["response"][0])
    #     print("system: ", outputs["system"][0])
    #     print("tools: ", outputs["tools"][0])
    #     print("images: ", outputs["images"][0])
    #     exit()

    prompt:  [{'role': 'user', 'content': '我们进行角色扮演，我将扮演一个催收公司的业务员，你将扮演一个债务人，对话的主要内容是我向你催债，你听明白了吗？'},
             {'role': 'assistant', 'content': '好的，我明白了，下面我将扮演债务人的角色。'},
             {'role': 'user', 'content': '啊。我这边是招商银行信用卡中心委托联信集团这边的。我姓林这边是有接到招商总行下发的这个材料。您这边是在招商银行，有这个信用卡没有处理好，是吧？'},
             {'role': 'assistant', 'content': '对你们有什么方案吗？'},
             {'role': 'user', 'content': '方案，是吧？你跟我说一下你的一个情况。现在的话，名下负债很多吗？'},
             {'role': 'assistant', 'content': '你们打了很多次电话，我每次都非常详细的跟你们说情况，然后过个几天，又换一个人打过来。我又再说一次，过几段时间又打过来。我又在说一次，那你们这样做事情有意思吗？'},
             {'role': 'user', 'content': '徐先生，现在这个材料是在我们这边。那我就简单问一下，你说负债是比较多的，对吗？'},
             {'role': 'assistant', 'content': '你是什么部门？'},
             {'role': 'user', 'content': '哦我这边是负责调解的。你问到方案的话，那如果说你名下负债比较多，不止招商一家的话，那我们可以提供到招商银行的政策的话，一个是可以通过减免这一块去处理。'},
             {'role': 'assistant', 'content': '你是什么公司还是什么。'},
             {'role': 'user', 'content': '我刚跟你说了这边的话是受招商银行信用卡中心委托联信集团那边的，我们是可以全权代表招商银行跟你沟通调解。'},
             {'role': 'assistant', 'content': '是我，那你是全权代表。我明白你，这你是什么公司？'},
             {'role': 'user', 'content': '啊联信集团。'},
             {'role': 'assistant', 'content': '是什么部门，'},
             {'role': 'user', 'content': '联信集团。'},
             {'role': 'assistant', 'content': '打两个字。'},
             {'role': 'user', 'content': '联信的联信用的信。'},
             {'role': 'assistant', 'content': '联信集团。'},
             {'role': 'user', 'content': '对。'},
             {'role': 'assistant', 'content': '你现在听到的情况是什么样子啊？'},
             {'role': 'user', 'content': '现在的话就说这边的话，按照招商银行现有的一个政策，对吧？这边的话是可以申请一个减免。'},
             {'role': 'assistant', 'content': '我先问你，你现在是在什么样的情况？我不是说现在有什么政策，我要知道你对这个事情了不了解？'},
             {'role': 'user', 'content': '徐先生，你材料刚下发到我们这边，你具体的一个情况这一块的话，我们也是不清楚你具体的一个情况。所以的话，既然说你想解决问题，我们也是来解助你解决问题的。'},
             {'role': 'assistant', 'content': '那我现在跟你说情况。好吧，因为我跟招商已经说了无数次了。然后我现在再跟你讲一次，我现在是在取保候审阶段。'},
             {'role': 'user', 'content': '嗯，嗯。哦。'},
             {'role': 'assistant', 'content': '所以我公司所有的钱都已经上交了，包括我自己所有账户的钱，也都已经交给公安了。现在是在取保候审阶段，现在是没有工作的。'},
             {'role': 'user', 'content': '嗯。哦。'},
             {'role': 'assistant', 'content': '我没取保候审。你也了解是什么情况嘛？对不对？是没有，没，没有办法去找工作。我现在只能等这个案子结束了之后，我再去找工作，然后来还这笔钱。我之前跟招商有提出过很多次协商方案，但是招商一直不同意，就说希望招商给我一个很长的一个分期，把金额每个月说小到一千块钱以内，或者一千块钱左右，我还。'}, {'role': 'user', 'content': '嗯。嗯。'}, {'role': 'assistant', 'content': '没想办法去，每个月去还。但是招商不同意。'}, {'role': 'user', 'content': '哦行那我明白嗯。'}, {'role': 'assistant', 'content': '明白我意思吗？所以他招商，招商给我的最好的一个方案就是让我每个月还最低还款额还四千多。我现在没有办法还这个钱。'}, {'role': 'user', 'content': '嗯。'}, {'role': 'assistant', 'content': '因为同时我在跟您说一下，就是我跟广发已经签了这个协议了。我欠广发的钱比欠招商的还有多。但是广发给我签了这个协议，我每个月只要还广发再来一千块钱左右，我是有在履行这个职责的。我跟平安也有协议，平安的话，每个月只还七百多。'}, {'role': 'user', 'content': '嗯。'}]
    response:  [{'role': 'assistant', 'content': '他们都了解情况下给我了这个方案。我都跟他们有好的去协商去解决这个事情。因为事情上，我现在取保候审阶段案子还没有看下来。我不是有钱，还是我所有的钱都已不上交了。就算我现在去打工，我也没有办法去打工。你明白吗？'}]
    system:
    tools:
    images:  []
    """

    return outputs


def convert_sharegpt(
    examples: Dict[str, List[Any]], dataset_attr: "DatasetAttr", data_args: "DataArguments"
) -> Dict[str, List[Any]]:
    r"""
    Converts sharegpt format dataset to the standard format.
    """
    outputs = {"prompt": [], "response": [], "system": [], "tools": [], "images": []}
    convert_images = partial(_convert_images, dataset_attr=dataset_attr, data_args=data_args)
    tag_mapping = {
        dataset_attr.user_tag: Role.USER.value,
        dataset_attr.assistant_tag: Role.ASSISTANT.value,
        dataset_attr.observation_tag: Role.OBSERVATION.value,
        dataset_attr.function_tag: Role.FUNCTION.value,
        dataset_attr.system_tag: Role.SYSTEM.value,
    }
    odd_tags = (dataset_attr.user_tag, dataset_attr.observation_tag)
    even_tags = (dataset_attr.assistant_tag, dataset_attr.function_tag)
    accept_tags = (odd_tags, even_tags)
    for i, messages in enumerate(examples[dataset_attr.messages]):
        if len(messages) == 0:
            continue

        if dataset_attr.system_tag and messages[0][dataset_attr.role_tag] == dataset_attr.system_tag:
            system = messages[0][dataset_attr.content_tag]
            messages = messages[1:]
        else:
            system = examples[dataset_attr.system][i] if dataset_attr.system else ""

        aligned_messages = []
        broken_data = False
        for turn_idx, message in enumerate(messages):
            if message[dataset_attr.role_tag] not in accept_tags[turn_idx % 2]:
                logger.warning("Invalid role tag in {}.".format(messages))
                broken_data = True

            aligned_messages.append(
                {"role": tag_mapping[message[dataset_attr.role_tag]], "content": message[dataset_attr.content_tag]}
            )

        if (not dataset_attr.ranking and len(aligned_messages) % 2 != 0) or (
            dataset_attr.ranking and len(aligned_messages) % 2 == 0
        ):
            logger.warning("Invalid message count in {}.".format(messages))
            broken_data = True

        if dataset_attr.kto_tag and isinstance(examples[dataset_attr.kto_tag][i], bool):  # kto example
            prompt = aligned_messages[:-1]
            response = aligned_messages[-1:]
            if examples[dataset_attr.kto_tag][i]:
                response = response + [{"role": Role.ASSISTANT.value, "content": ""}]
            else:
                response = [{"role": Role.ASSISTANT.value, "content": ""}] + response
        elif (
            dataset_attr.ranking
            and isinstance(examples[dataset_attr.chosen][i], dict)
            and isinstance(examples[dataset_attr.rejected][i], dict)
        ):  # pairwise example
            chosen = examples[dataset_attr.chosen][i]
            rejected = examples[dataset_attr.rejected][i]
            if (
                chosen[dataset_attr.role_tag] not in accept_tags[-1]
                or rejected[dataset_attr.role_tag] not in accept_tags[-1]
            ):
                logger.warning("Invalid role tag in {}.".format([chosen, rejected]))
                broken_data = True

            prompt = aligned_messages
            response = [
                {"role": tag_mapping[chosen[dataset_attr.role_tag]], "content": chosen[dataset_attr.content_tag]},
                {"role": tag_mapping[rejected[dataset_attr.role_tag]], "content": rejected[dataset_attr.content_tag]},
            ]
        else:  # normal example
            prompt = aligned_messages[:-1]
            response = aligned_messages[-1:]

        if broken_data:
            logger.warning("Skipping this abnormal example.")
            continue

        outputs["prompt"].append(prompt)
        outputs["response"].append(response)
        outputs["system"].append(system)
        outputs["tools"].append(examples[dataset_attr.tools][i] if dataset_attr.tools else "")
        outputs["images"].append(convert_images(examples[dataset_attr.images][i]) if dataset_attr.images else [])

    return outputs


def align_dataset(
    dataset: Union["Dataset", "IterableDataset"],
    dataset_attr: "DatasetAttr",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
) -> Union["Dataset", "IterableDataset"]:
    r"""
    Aligned dataset:
        prompt: [{"role": "user", "content": "..."}] * (2T - 1)
        response: [{"role": "assistant", "content": "..."}] * N (N > 1 for ranking dataset)
        system: "..."
        tools: "...",
        images: [],
    """
    if dataset_attr.formatting == "alpaca":
        # 添加对 channel loss 的支持，数据额外加一个 channel 字段   (只改了 alpaca 数据格式)
        convert_func = partial(convert_alpaca, dataset_attr=dataset_attr, data_args=data_args)  # 默认
    else:
        convert_func = partial(convert_sharegpt, dataset_attr=dataset_attr, data_args=data_args)

    column_names = list(next(iter(dataset)).keys())
    features = Features.from_dict(
        {
            "prompt": [
                {"role": {"dtype": "string", "_type": "Value"}, "content": {"dtype": "string", "_type": "Value"}}
            ],
            "response": [
                {"role": {"dtype": "string", "_type": "Value"}, "content": {"dtype": "string", "_type": "Value"}}
            ],
            "system": {"dtype": "string", "_type": "Value"},
            "tools": {"dtype": "string", "_type": "Value"},
            "images": [{"_type": "Image"}],
        }
    )

    # TODO: 【√】添加对 channel loss 的支持，数据额外加一个 channel 字段   (只改了 alpaca 数据格式)
    if data_args.channel_loss:
        features["channels"] = [Value("int32")]
        print("[DEBUG] use channel loss, dataset_name: {}, features: {}.".format(dataset_attr.dataset_name.split(".json")[0], features))

    kwargs = {}
    if not data_args.streaming:
        kwargs = dict(
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=(not data_args.overwrite_cache) or (training_args.local_process_index != 0),
            desc="Converting format of dataset",
        )
    # e.g. kwargs:  {'num_proc': 16, 'load_from_cache_file': False, 'desc': 'Converting format of dataset'}

    return dataset.map(
        convert_func,   # 批处理样本
        batched=True,
        remove_columns=column_names,
        features=features,
        **kwargs,
    )
