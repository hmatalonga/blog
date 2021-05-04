<template>
  <div>
    <div class="mb-8 md:mb-10">
      <h1
        class="font-medium leading-tight text-gray-800 text-3xl md:text-4xl lg:text-5xl"
      >
        {{ page.title }}
      </h1>
      <p class="text-gray-600 text-base md:text-lg my-3">{{ timestamp }}</p>
    </div>
    <nuxt-content
      class="prose sm:prose-sm lg:prose-lg max-w-none"
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
      title: this.page.title,
      link: [
        {
          rel: 'canonical',
          href: `https://hmatalonga.com${this.$route.path}`,
        },
      ],
      meta: [
        {
          hid: 'description',
          name: 'description',
          content: this.page.description,
        },
        // Open Graph
        { hid: 'og:title', property: 'og:title', content: this.page.title },
        {
          hid: 'og:description',
          property: 'og:description',
          content: this.page.description,
        },
        {
          hid: 'og:url',
          property: 'og:url',
          content: `https://hmatalonga.com${this.$route.path}`,
        },
        // Twitter Card
        {
          hid: 'twitter:title',
          name: 'twitter:title',
          content: this.page.title,
        },
        {
          hid: 'twitter:description',
          name: 'twitter:description',
          content: this.page.description,
        },
      ],
    }
  },
}
</script>
