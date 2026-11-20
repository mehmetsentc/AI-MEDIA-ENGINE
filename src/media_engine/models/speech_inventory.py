"""Pinned model file inventories for staging.

TARGETS is the speech session's pinned pair and nothing else belongs in it.
Two safety values are derived from it and both break if it grows: the
authorization pins exactly one model per capability, so a second audio.tts
entry reads as model substitution, and the staging disk budget is the sum of
every tree in it, so extra entries demand disk a run never uses.

EXTRA_TARGETS carries every other model that can be staged. Nothing derives a
budget or a pin from it; it exists so a tree can be fetched and verified.

sha_source records how a checksum was obtained, and the three are not equally
strong:

lfs        the Git LFS object id published for that revision. SHA-256, but the
           bytes were never downloaded, so this is the hub's claim.
content    hashed from a file small enough to read at pinning time.
downloaded the whole file was fetched and hashed. The strongest of the three:
           these checksums describe bytes that were actually held.
"""
from __future__ import annotations

TARGETS: tuple[dict, ...] = (
    {
        'model_id': 'Systran/faster-whisper-large-v3',
        'revision': 'edaa852ec7e145841d8ffdb056a99866b5f0a478',
        'capability': 'audio.transcribe',
        'hub_used_storage': 3145762253,
        'files': (
            {'path': '.gitattributes', 'bytes': 1519, 'sha256': '11ad7efa24975ee4b0c3c3a38ed18737f0658a5f75a0a96787b576a78a023361', 'sha_source': 'content'},
            {'path': 'README.md', 'bytes': 2052, 'sha256': '39e96252229f5a3d0141dc81afb65a36fd205461ac21e5b70f2cd1248ef0082c', 'sha_source': 'content'},
            {'path': 'config.json', 'bytes': 2394, 'sha256': 'a9306624f5ec14270a014b647e5c316b6e03a662c369758d1b90697a7b0655b9', 'sha_source': 'content'},
            {'path': 'model.bin', 'bytes': 3087284237, 'sha256': '69f74147e3334731bc3a76048724833325d2ec74642fb52620eda87352e3d4f1', 'sha_source': 'lfs'},
            {'path': 'preprocessor_config.json', 'bytes': 340, 'sha256': '7ccc62c6f2765af1f3b46c00c9b5894426835a05021c8b9c01eecb6dfb542711', 'sha_source': 'content'},
            {'path': 'tokenizer.json', 'bytes': 2480617, 'sha256': '6d8cbd7cd0d8d5815e478dac67b85a26bbe77c1f5e0c6d76d1ce2abc0e5f21ca', 'sha_source': 'content'},
            {'path': 'vocabulary.json', 'bytes': 1068114, 'sha256': 'c69260f2ab26d659b7c398f9a2b2b48ed0df16c3b47d7326782fd9cba71690c1', 'sha_source': 'content'},
        ),
    },
    {
        'model_id': 'Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice',
        'revision': '0c0e3051f131929182e2c023b9537f8b1c68adfe',
        'capability': 'audio.tts',
        'hub_used_storage': 4617449338,
        'files': (
            {'path': '.gitattributes', 'bytes': 1519, 'sha256': '11ad7efa24975ee4b0c3c3a38ed18737f0658a5f75a0a96787b576a78a023361', 'sha_source': 'content'},
            {'path': 'README.md', 'bytes': 57846, 'sha256': '64f65e809b51cc0c35f393fbbfcc2d735c0cb3fdbbc6f3fdc4a5e6ce55e9d088', 'sha_source': 'content'},
            {'path': 'config.json', 'bytes': 4908, 'sha256': '17a07f527a1c25ea30b4e023a184482a23d3e279d697b1dc81b1bde498d29cf9', 'sha_source': 'content'},
            {'path': 'generation_config.json', 'bytes': 245, 'sha256': 'f1b90b4513f3b34c62851049e2492d7b4c5940daf1276f89c82b8ef04127f3aa', 'sha_source': 'content'},
            {'path': 'merges.txt', 'bytes': 1671839, 'sha256': '599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3', 'sha_source': 'content'},
            {'path': 'model.safetensors', 'bytes': 3833402552, 'sha256': '38b1d5971bdbd982b561cccec982669a53b0537c3cf5e9bd4778ed07bb2f5137', 'sha_source': 'lfs'},
            {'path': 'preprocessor_config.json', 'bytes': 127, 'sha256': 'efdde1022ea9d76928bf7a9cd53139138f5ba2e466e837f08f6105ab1af1c119', 'sha_source': 'content'},
            {'path': 'speech_tokenizer/config.json', 'bytes': 2336, 'sha256': 'ee65bb901c876664ab8707c487157aa1a6ee57c65969b28fb5ec9dc211e68167', 'sha_source': 'content'},
            {'path': 'speech_tokenizer/configuration.json', 'bytes': 76, 'sha256': '6bc26d64eb5024b4d1dab5a52371958b429256d6c9d59787f1f5294a54e0cebd', 'sha_source': 'content'},
            {'path': 'speech_tokenizer/model.safetensors', 'bytes': 682293092, 'sha256': '836b7b357f5ea43e889936a3709af68dfe3751881acefe4ecf0dbd30ba571258', 'sha_source': 'lfs'},
            {'path': 'speech_tokenizer/preprocessor_config.json', 'bytes': 234, 'sha256': 'fcb3805e597e786d4067706e602f6688524640f8d3396790e2e09b5942fcbdfb', 'sha_source': 'content'},
            {'path': 'tokenizer_config.json', 'bytes': 7344, 'sha256': 'dc3c31c3bdaedd5016382bb3cbe07323026775ad51f5a4fb564505992ae4a670', 'sha_source': 'content'},
            {'path': 'vocab.json', 'bytes': 2776833, 'sha256': 'ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910', 'sha_source': 'content'},
        ),
    },
)


EXTRA_TARGETS: tuple[dict, ...] = (
    {
        'model_id': 'freyavoice/Freya-TTS',
        'revision': 'd124e07493615208f58bdd21d432736849ee4230',
        'capability': 'audio.tts',
        'hub_used_storage': 732812048,
        'files': (
            {'path': 'config.json', 'bytes': 92, 'sha256': '898c9a951a20f960e0936d4e222c2a4044d383f5691efa1a614d812812956a67', 'sha_source': 'downloaded'},
            {'path': 'model.safetensors', 'bytes': 732811956, 'sha256': '9e5828ce9eb6aaf197adc5cb098e2e80e8ff301d238add631418c5557fa08b22', 'sha_source': 'downloaded'},
        ),
    },
    {
        'model_id': 'Qwen/Qwen3-TTS-12Hz-1.7B-Base',
        'revision': 'fd4b254389122332181a7c3db7f27e918eec64e3',
        'capability': 'audio.voice_clone',
        'hub_used_storage': 4544229700,
        'files': (
            {'path': '.gitattributes', 'bytes': 1519, 'sha256': '11ad7efa24975ee4b0c3c3a38ed18737f0658a5f75a0a96787b576a78a023361', 'sha_source': 'downloaded'},
            {'path': 'README.md', 'bytes': 57817, 'sha256': '621b7e88f1867bc03d817176fc1f7f55f4b5a70654b2a07fdcda1e169efb024b', 'sha_source': 'downloaded'},
            {'path': 'config.json', 'bytes': 4494, 'sha256': 'b4f01752d15a488abde3e1ab44723ae4f4b9e68a4037257b098b3737893cc1f9', 'sha_source': 'downloaded'},
            {'path': 'generation_config.json', 'bytes': 245, 'sha256': 'f1b90b4513f3b34c62851049e2492d7b4c5940daf1276f89c82b8ef04127f3aa', 'sha_source': 'downloaded'},
            {'path': 'merges.txt', 'bytes': 1671839, 'sha256': '599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3', 'sha_source': 'downloaded'},
            {'path': 'model.safetensors', 'bytes': 3857413744, 'sha256': '38fc7fc51c5e776e840414b6fd443962e9411b9654888fd7913e4da643cb857c', 'sha_source': 'downloaded'},
            {'path': 'preprocessor_config.json', 'bytes': 127, 'sha256': 'efdde1022ea9d76928bf7a9cd53139138f5ba2e466e837f08f6105ab1af1c119', 'sha_source': 'downloaded'},
            {'path': 'speech_tokenizer/config.json', 'bytes': 2336, 'sha256': 'ee65bb901c876664ab8707c487157aa1a6ee57c65969b28fb5ec9dc211e68167', 'sha_source': 'downloaded'},
            {'path': 'speech_tokenizer/configuration.json', 'bytes': 76, 'sha256': '6bc26d64eb5024b4d1dab5a52371958b429256d6c9d59787f1f5294a54e0cebd', 'sha_source': 'downloaded'},
            {'path': 'speech_tokenizer/model.safetensors', 'bytes': 682293092, 'sha256': '836b7b357f5ea43e889936a3709af68dfe3751881acefe4ecf0dbd30ba571258', 'sha_source': 'downloaded'},
            {'path': 'speech_tokenizer/preprocessor_config.json', 'bytes': 234, 'sha256': 'fcb3805e597e786d4067706e602f6688524640f8d3396790e2e09b5942fcbdfb', 'sha_source': 'downloaded'},
            {'path': 'tokenizer_config.json', 'bytes': 7344, 'sha256': 'dc3c31c3bdaedd5016382bb3cbe07323026775ad51f5a4fb564505992ae4a670', 'sha_source': 'downloaded'},
            {'path': 'vocab.json', 'bytes': 2776833, 'sha256': 'ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910', 'sha_source': 'downloaded'},
        ),
    },
    {
        'model_id': 'Qwen/Qwen2.5-1.5B-Instruct',
        'revision': '989aa7980e4cf806f80c7fef2b1adb7bc71aa306',
        'capability': 'audio.translate',
        'hub_used_storage': 3098967011,
        'files': (
            {'path': 'LICENSE', 'bytes': 11343, 'sha256': '832dd9e00a68dd83b3c3fb9f5588dad7dcf337a0db50f7d9483f310cd292e92e', 'sha_source': 'downloaded'},
            {'path': 'config.json', 'bytes': 660, 'sha256': '98d2ff8cc47488d08a2b0b3acf4eb99ef210779b42bd48605f6b8e36acdbf670', 'sha_source': 'downloaded'},
            {'path': 'generation_config.json', 'bytes': 242, 'sha256': 'e558847a8b4402616f1273797b015104dc266fe4b520056fca88823ba8f8ebe6', 'sha_source': 'downloaded'},
            {'path': 'merges.txt', 'bytes': 1671839, 'sha256': '599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3', 'sha_source': 'downloaded'},
            {'path': 'model.safetensors', 'bytes': 3087467144, 'sha256': 'dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee', 'sha_source': 'downloaded'},
            {'path': 'tokenizer.json', 'bytes': 7031645, 'sha256': 'c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539', 'sha_source': 'downloaded'},
            {'path': 'tokenizer_config.json', 'bytes': 7305, 'sha256': '5b5d4f65d0acd3b2d56a35b56d374a36cbc1c8fa5cf3b3febbbfabf22f359583', 'sha_source': 'downloaded'},
            {'path': 'vocab.json', 'bytes': 2776833, 'sha256': 'ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910', 'sha_source': 'downloaded'},
        ),
    },
    {
        'model_id': 'openbmb/VoxCPM2',
        'revision': '7936080b41163874f38610f46f438e2c0757f9fd',
        'capability': 'audio.voice_clone',
        'hub_used_storage': 4960730311,
        'files': (
            {'path': 'README.md', 'bytes': 7903, 'sha256': 'feb850b390aa5051b81cbdc73b048361273552f003dd386b4da9c39ec9b6f055', 'sha_source': 'downloaded'},
            {'path': 'audiovae.pth', 'bytes': 376951122, 'sha256': '94b5d51e107e0507d4acc976cfdadb64edd6fd06d1f751dadbf2fd1594274bf1', 'sha_source': 'downloaded'},
            {'path': 'config.json', 'bytes': 4336, 'sha256': '405f0dcd92f7feba6011ed4eac5c8d4f74cba9712f07fd5cfa3063bbdd95402c', 'sha_source': 'downloaded'},
            {'path': 'model.safetensors', 'bytes': 4580080592, 'sha256': 'f7f964cfa9da23653baec6e6f7750719977ad944ed9f95fe52fe3a620506891d', 'sha_source': 'downloaded'},
            {'path': 'special_tokens_map.json', 'bytes': 1632, 'sha256': '068594063e37662c02b21acf42ebb334ef6a74fb810e68a2368f88f08351de76', 'sha_source': 'downloaded'},
            {'path': 'tokenization_voxcpm2.py', 'bytes': 2895, 'sha256': '84489ea32b6ee0cae22ed5480cacb6df85c46624c3119be9a2021c3649a12729', 'sha_source': 'downloaded'},
            {'path': 'tokenizer.json', 'bytes': 3676772, 'sha256': 'f8984687e4a92a3503d521396d454b7d68e9fdaab2a0288eb3536c7c1aa4bc20', 'sha_source': 'downloaded'},
            {'path': 'tokenizer_config.json', 'bytes': 5059, 'sha256': 'e78a3ebb48a0b9437efd1823b6b726c823da89e49dd8bcc90c02419d9baa772b', 'sha_source': 'downloaded'},
        ),
    },
)
