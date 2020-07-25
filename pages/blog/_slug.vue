<template>
  <div>
    <div class="mb-8 md:mb-10">
      <h1
        class="font-semibold leading-tight text-gray-800 text-3xl md:text-4xl lg:text-5xl"
      >
        {{ page.title }}
      </h1>
      <p class="text-gray-600 text-base md:text-lg my-3">{{ timestamp }}</p>
    </div>
    <nuxt-content
      class="prose prose-sm sm:prose-sm lg:prose-lg xl:prose-xl max-w-none"
      :document="page"
    />
    <Signature />
  </div>
</template>

<script>
import tinytime from 'tinytime'
import Signature from '~/components/Signature.vue'

export default {
  components: {
    Signature,
  },
  async asyncData({ $content, params, error }) {
    const slug = `blog/${params.slug}` || 'index'
    const page = await $content(slug)
      .fetch()
      .catch(() => {
        error({ statusCode: 404, message: 'Page not found' })
      })

    return {
      page,
      timestamp: tinytime('{MM} {DD}, {YYYY}').render(new Date(page.date)),
    }
  },
  head() {
    return {
      title: `${this.page.title} | Hugo Matalonga`,
    }
  },
}
</script>
