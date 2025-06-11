
<template>

    <input type="file" accept="application/pdf" @change="handleFileUpload" />
    <button @click="submit">Upload PDF</button>
    <p>{{ status }}</p>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import axios from 'axios'

const file = ref(null)
const status = ref('')

const handleFileUpload = (event) => {
  file.value = event.target.files[0]
}

const submit = async () => {
  if (!file.value) {
    status.value = "Please select a file first."
    return
  }

  const formData = new FormData()
  formData.append('file', file.value)

  try {
    const response = await axios.post('http://localhost:5000/api/pdf/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data'
      }
    })
    status.value = response.data.message
  } catch (err) {
    status.value = err.response?.data?.error || 'Upload failed'
  }
}

</script>


<style scoped>
.logo {
  height: 6em;
  padding: 1.5em;
  will-change: filter;
  transition: filter 300ms;
}
.logo:hover {
  filter: drop-shadow(0 0 2em #646cffaa);
}
.logo.vue:hover {
  filter: drop-shadow(0 0 2em #42b883aa);
}
</style>
