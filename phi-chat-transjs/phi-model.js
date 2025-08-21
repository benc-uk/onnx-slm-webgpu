import { checkCache } from '../lib/utils.js'
import { addErrorMsg, addStatusMsg, showQueryControls, setResponseText } from './ui.js'

// *************************************************************************************
// Only works with local copy of the model until this PR is merged:
// https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-onnx-web/discussions/3
// *************************************************************************************

// Using a local patched build v3 of the Transformers library with use_external_data_format bug fixed
// Need until this MR is merged https://github.com/huggingface/transformers.js/pull/1180
import { env, pipeline, TextStreamer } from '/home/ben/temp/transformers.js/dist/transformers.js'
//import { env, pipeline, TextStreamer } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.3.3'

const MODEL = 'microsoft/Phi-3-mini-4k-instruct-onnx-web'
const USE_LOCAL_MODEL = true

env.localModelPath = '../models'
env.allowRemoteModels = !USE_LOCAL_MODEL
env.allowLocalModels = USE_LOCAL_MODEL

let phiPipeline = null

export async function setUp() {
  addStatusMsg(`🧠 Loading model... `)

  const inCache = await checkCache(`model_q4f16.onnx`, 'transformers-cache')
  if (!inCache) {
    addStatusMsg(`⌛ Downloading model from ${USE_LOCAL_MODEL ? 'localhost' : 'huggingface'}...`)
    addStatusMsg(`⌛ Warning: This is 2GB of data! please be patient...`)
  } else {
    addStatusMsg(`📦 Model was found in cache`)
  }

  let oldProgress = 0
  try {
    phiPipeline = await pipeline('text-generation', MODEL, {
      device: 'webgpu',
      use_external_data_format: true,
      dtype: 'q4f16',
      // session_options: {
      //   use_external_data_format: true,
      // },
      progress_callback: (data) => {
        const progFloor = Math.floor(data.progress)
        if (progFloor % 5 === 0 && oldProgress !== progFloor) {
          oldProgress = progFloor
          addStatusMsg(`💾 Download ${data.file} - ${progFloor}%`)
        }
      },
    })

    addStatusMsg('🚀 Model loaded, session started!')
    showQueryControls()
  } catch (e) {
    addErrorMsg('' + e)
  }
}

export async function queryModel(query, id, continuation = false) {
  addStatusMsg(`🧠 Beginning query for "${query}"`)

  try {
    let responseText = ''

    // Use the TextStreamer to handle the output
    const streamer = new TextStreamer(phiPipeline.tokenizer, {
      skip_prompt: true,
      callback_function: (text) => {
        responseText += text
        setResponseText(id, responseText)
      },
    })

    const messages = [
      { role: 'system', content: 'You are a helpful assistant.' },
      { role: 'user', content: query },
    ]

    await phiPipeline(messages, { max_new_tokens: 1024, do_sample: false, streamer })
  } catch (e) {
    addErrorMsg('' + e)
  }
}
